#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#

'''
Hybrid density fitting

Divide the AO basis into two parts, compact (c) and diffuse (d). Compute
(P|cc) and (P|cd) in real space using normal GDF, which allow access to
ERIs of type (cc|cc), (cc|cd), (cd|cd). The remaining types of ERIs that
involve the "dd"-type shell pair, i.e., (cc|dd), (cd|dd), and (dd|dd), are
computed in reciprocal space using analytical Fourial transform (AFT).
'''

import os
import sys
import time
import h5py
import copy
import tempfile
import threading
import contextlib
import numpy as np
import scipy.special

from pyscf import gto as mol_gto
from pyscf.gto.mole import PTR_RANGE_OMEGA
from pyscf.pbc import df
from pyscf.pbc.df import outcore
from pyscf.pbc.df import rshdf_ao2mo
from pyscf.pbc.tools import k2gamma
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.pbc import tools as pbctools
from pyscf.pbc.lib.kpts_helper import (is_zero, gamma_point, member, unique,
                                       KPT_DIFF_TOL)
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__

LINEAR_DEP_THR = getattr(__config__, 'pbc_df_df_DF_lindep', 1e-9)
LONGRANGE_AFT_TURNOVER_THRESHOLD = 2.5



def get_aux_chg(auxcell):
    def get_nd(l):
        if auxcell.cart:
            return (l+1) * (l+2) // 2
        else:
            return 2 * l + 1

    naux = auxcell.nao_nr()
    qs = np.zeros(naux)
    shift = 0
    half_sph_norm = np.sqrt(4*np.pi)
    for ib in range(auxcell.nbas):
        l = auxcell.bas_angular(ib)
        if l == 0:
            npm = auxcell.bas_nprim(ib)
            nc = auxcell.bas_nctr(ib)
            es = auxcell.bas_exp(ib)
            ptr = auxcell._bas[ib,mol_gto.PTR_COEFF]
            cs = auxcell._env[ptr:ptr+npm*nc].reshape(nc,npm).T
            norms = mol_gto.gaussian_int(l+2, es)
            q = np.einsum("i,ij->j",norms,cs)[0] * half_sph_norm
        else:   # higher angular momentum AOs carry no charge
            q = 0.
        nd = get_nd(l)
        qs[shift:shift+nd] = q
        shift += nd

    return qs


# kpti == kptj: s2 symmetry
# kpti == kptj == 0 (gamma point): real
def _make_j3c(mydf, cell, auxcell, cell_fat, kptij_lst, cderi_file):
    t1 = (time.clock(), time.time())
    log = logger.Logger(mydf.stdout, mydf.verbose)
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])

    if mydf.use_bvkcell:
        bvk_kmesh = k2gamma.kpts_to_kmesh(cell, mydf.kpts)
    else:
        bvk_kmesh = None

    # The ideal way to hold the temporary integrals is to store them in the
    # cderi_file and overwrite them inplace in the second pass.  The current
    # HDF5 library does not have an efficient way to manage free space in
    # overwriting.  It often leads to the cderi_file ~2 times larger than the
    # necessary size.  For now, dumping the DF integral intermediates to a
    # separated temporary file can avoid this issue.  The DF intermediates may
    # be terribly huge. The temporary file should be placed in the same disk
    # as cderi_file.
    swapfile = tempfile.NamedTemporaryFile(dir=os.path.dirname(cderi_file))
    fswap = lib.H5TmpFile(swapfile.name)
    # Unlink swapfile to avoid trash
    swapfile = None

    # get charge of auxbasis
    qaux = get_aux_chg(auxcell)
    g0 = np.pi/mydf.omega**2./cell.vol

    # @@HY: compute j2c first as it informs the integral screening
    nao = cell.nao_nr()
    naux = auxcell.nao_nr()

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)

    log.debug('Num uniq kpts %d', len(uniq_kpts))
    log.debug2('uniq_kpts %s', uniq_kpts)
    # j2c ~ (-kpt_ji | kpt_ji)
    j2c = auxcell.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)

    # Add charge contribution for j2c
    qaux2 = None
    for k, kpt in enumerate(uniq_kpts):
        if is_zero(kpt):
            if qaux2 is None:
                qaux2 = np.outer(qaux,qaux)
            j2c[k] -= qaux2 * g0
        fswap['j2c/%d'%k] = j2c[k]
    j2c = qaux2 = None

    t1 = log.timer_debug1('2c2e', *t1)

    def cholesky_decomposed_metric(uniq_kptji_id):
        j2c = np.asarray(fswap['j2c/%d'%uniq_kptji_id])
        j2c_negative = None
        try:
            j2c = scipy.linalg.cholesky(j2c, lower=True)
            j2ctag = 'CD'
        except scipy.linalg.LinAlgError:
            #msg =('===================================\n'
            #      'J-metric not positive definite.\n'
            #      'It is likely that mesh is not enough.\n'
            #      '===================================')
            #log.error(msg)
            #raise scipy.linalg.LinAlgError('\n'.join([str(e), msg]))
            w, v = scipy.linalg.eigh(j2c)
            log.debug('DF metric linear dependency for kpt %s', uniq_kptji_id)
            log.debug('cond = %.4g, drop %d bfns',
                      w[-1]/w[0], np.count_nonzero(w<mydf.linear_dep_threshold))
            v1 = v[:,w>mydf.linear_dep_threshold].conj().T
            v1 /= np.sqrt(w[w>mydf.linear_dep_threshold]).reshape(-1,1)
            j2c = v1
            if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
                idx = np.where(w < -mydf.linear_dep_threshold)[0]
                if len(idx) > 0:
                    j2c_negative = (v[:,idx]/np.sqrt(-w[idx])).conj().T
            w = v = None
            j2ctag = 'eig'
        return j2c, j2c_negative, j2ctag

    # compute j3c
    k_gamma = None
    for k, kpt in enumerate(uniq_kpts):
        if is_zero(kpt):
            k_gamma = k
            break
    if k_gamma is None: k_gamma = 0
    j2c, j2c_negative, j2ctag = cholesky_decomposed_metric(k_gamma)
    if j2ctag == "CD":
        j2c_inv = np.eye(j2c.shape[0])
        j2c_inv = scipy.linalg.solve_triangular(j2c, j2c_inv, lower=True,
                                                overwrite_b=True)
    else:
        j2c_inv = j2c
    extra_precision = 1./(np.max(np.abs(j2c_inv), axis=0)+1.)

    prescreening_data = get_prescreening_data(mydf, cell_fat, extra_precision)

    t1 = log.timer_debug1('prescrn warmup', *t1)

    outcore._aux_e2_hy(cell, auxcell, fswap, 'int3c2e', aosym='s2',
                       kptij_lst=kptij_lst, dataname='j3c-junk',
                       max_memory=max_memory,
                       bvk_kmesh=bvk_kmesh,
                       prescreening_type=mydf.prescreening_type,
                       prescreening_data=prescreening_data,
                       cell_fat=cell_fat)
    t1 = log.timer_debug1('3c2e', *t1)

    feri = h5py.File(cderi_file, 'w')
    feri['j3c-kptij'] = kptij_lst
    nsegs = len(fswap['j3c-junk/0'])
    def make_kpt(uniq_kptji_id, cholesky_j2c):
        kpt = uniq_kpts[uniq_kptji_id]  # kpt = kptj - kpti
        log.debug1('kpt = %s', kpt)
        adapted_ji_idx = np.where(uniq_inverse == uniq_kptji_id)[0]
        adapted_kptjs = kptjs[adapted_ji_idx]
        nkptj = len(adapted_kptjs)
        log.debug1('adapted_ji_idx = %s', adapted_ji_idx)

        j2c, j2c_negative, j2ctag = cholesky_j2c

        if is_zero(kpt):  # kpti == kptj
            aosym = 's2'
            nao_pair = nao*(nao+1)//2

            if cell.dimension == 3:
                vbar = qaux * g0
                if cell_fat is None:
                    ovlp = cell.pbc_intor('int1e_ovlp', hermi=1,
                                          kpts=adapted_kptjs)
                    ovlp = [lib.pack_tril(s) for s in ovlp]
                else:
                    nao_fat = cell_fat.nao_nr()
                    ovlp_fat = np.asarray(cell_fat.pbc_intor('int1e_ovlp',
                                          hermi=1, kpts=adapted_kptjs)).real.reshape(
                                          -1,nao_fat*nao_fat)
                    ovlp = np.zeros((ovlp_fat.shape[0],nao*nao),
                                    dtype=ovlp_fat.dtype)
                    for iap_fat, iap in rshdf_ao2mo.fat_orig_loop(
                                                            cell_fat, cell,
                                                            aosym='s1',
                                                            sr_only=True):
                        ovlp[:,iap] += ovlp_fat[:,iap_fat]
                    ovlp = [lib.pack_tril(s.reshape(nao,nao)) for s in ovlp]
        else:
            aosym = 's1'
            nao_pair = nao**2

        mem_now = lib.current_memory()[0]
        log.debug2('memory = %s', mem_now)
        max_memory = max(2000, mydf.max_memory-mem_now)
        # nkptj for 3c-coulomb arrays plus 1 Lpq array
        buflen = min(max(int(max_memory*.38e6/16/naux/(nkptj+1)), 1), nao_pair)
        shranges = _guess_shell_ranges(cell, buflen, aosym)
        buflen = max([x[2] for x in shranges])
        def j3c_contract(istep, j3cR, j3cI):
            for k, ji in enumerate(adapted_ji_idx):
                if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                    v = j3cR[k]
                else:
                    v = j3cR[k] + j3cI[k] * 1j
                if j2ctag == 'CD':
                    v = scipy.linalg.solve_triangular(j2c, v, lower=True, overwrite_b=True)
                    feri['j3c/%d/%d'%(ji,istep)] = v
                else:
                    feri['j3c/%d/%d'%(ji,istep)] = lib.dot(j2c, v)

                # low-dimension systems
                if j2c_negative is not None:
                    feri['j3c-/%d/%d'%(ji,istep)] = lib.dot(j2c_negative, v)

        with lib.call_in_background(j3c_contract) as compute:
            col1 = 0
            for istep, sh_range in enumerate(shranges):
                log.debug1('int3c2e [%d/%d], AO [%d:%d], ncol = %d',
                           istep+1, len(shranges), *sh_range)
                bstart, bend, ncol = sh_range
                col0, col1 = col1, col1+ncol
                j3cR = []
                j3cI = []
                for k, idx in enumerate(adapted_ji_idx):
                    v = np.vstack([fswap['j3c-junk/%d/%d'%(idx,i)][0,col0:col1].T
                                      for i in range(nsegs)])
                    # vbar is the interaction between the background charge
                    # and the auxiliary basis.  0D, 1D, 2D do not have vbar.
                    if is_zero(kpt) and cell.dimension == 3:
                        for i in np.where(vbar != 0)[0]:
                            v[i] -= vbar[i] * ovlp[k][col0:col1]
                    j3cR.append(np.asarray(v.real, order='C'))
                    if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                        j3cI.append(None)
                    else:
                        j3cI.append(np.asarray(v.imag, order='C'))
                v = None
                compute(istep, j3cR, j3cI)
        for ji in adapted_ji_idx:
            del(fswap['j3c-junk/%d'%ji])

    # Wrapped around boundary and symmetry between k and -k can be used
    # explicitly for the metric integrals.  We consider this symmetry
    # because it is used in the df_ao2mo module when contracting two 3-index
    # integral tensors to the 4-index 2e integral tensor. If the symmetry
    # related k-points are treated separately, the resultant 3-index tensors
    # may have inconsistent dimension due to the numerial noise when handling
    # linear dependency of j2c.
    def conj_j2c(cholesky_j2c):
        j2c, j2c_negative, j2ctag = cholesky_j2c
        if j2c_negative is None:
            return j2c.conj(), None, j2ctag
        else:
            return j2c.conj(), j2c_negative.conj(), j2ctag

    a = cell.lattice_vectors() / (2*np.pi)
    def kconserve_indices(kpt):
        '''search which (kpts+kpt) satisfies momentum conservation'''
        kdif = np.einsum('wx,ix->wi', a, uniq_kpts + kpt)
        kdif_int = np.rint(kdif)
        mask = np.einsum('wi->i', abs(kdif - kdif_int)) < KPT_DIFF_TOL
        uniq_kptji_ids = np.where(mask)[0]
        return uniq_kptji_ids

    done = np.zeros(len(uniq_kpts), dtype=bool)
    for k, kpt in enumerate(uniq_kpts):
        if done[k]:
            continue

        log.debug1('Cholesky decomposition for j2c at kpt %s', k)
        cholesky_j2c = cholesky_decomposed_metric(k)

        # The k-point k' which has (k - k') * a = 2n pi. Metric integrals have the
        # symmetry S = S
        uniq_kptji_ids = kconserve_indices(-kpt)
        log.debug1("Symmetry pattern (k - %s)*a= 2n pi", kpt)
        log.debug1("    make_kpt for uniq_kptji_ids %s", uniq_kptji_ids)
        for uniq_kptji_id in uniq_kptji_ids:
            if not done[uniq_kptji_id]:
                make_kpt(uniq_kptji_id, cholesky_j2c)
        done[uniq_kptji_ids] = True

        # The k-point k' which has (k + k') * a = 2n pi. Metric integrals have the
        # symmetry S = S*
        uniq_kptji_ids = kconserve_indices(kpt)
        log.debug1("Symmetry pattern (k + %s)*a= 2n pi", kpt)
        log.debug1("    make_kpt for %s", uniq_kptji_ids)
        cholesky_j2c = conj_j2c(cholesky_j2c)
        for uniq_kptji_id in uniq_kptji_ids:
            if not done[uniq_kptji_id]:
                make_kpt(uniq_kptji_id, cholesky_j2c)
        done[uniq_kptji_ids] = True

    feri.close()


class RangeSeparatedHybridDensityFitting(df.df.GDF):
    '''Range Separated Hybrid Density Fitting
    '''

    # class methods defined outside the class
    _make_j3c = _make_j3c

    # ao2mo
    get_eri = get_ao_eri = rshdf_ao2mo.get_eri
    # ao2mo = get_mo_eri = rshdf_ao2mo.general
    # ao2mo_7d = rshdf_ao2mo.ao2mo_7d

    def __init__(self, cell, kpts=np.zeros((1,3))):
        df.df.GDF.__init__(self, cell, kpts=kpts)

        self.use_bvkcell = False    # for testing
        self.prescreening_type = 0
        self.split_basis = False

        self.kpts = kpts  # default is gamma point
        self.kpts_band = None
        self._auxbasis = None

        # One of {omega, ke_cutoff} must be provided, and the other will be
        # deduced automatically. If both are provided, omega will be used.
        self.omega = None
        self.npw_max = 600
        self.ke_cutoff = None
        self.mesh_sr = None

        # If split_basis is True, each ao shell will be split into a diffuse (d)
        # part and a compact (c) part based on the pGTO exponents.
        # If eta_lr is given, eta_lr will be used.
        # If eta_lr is not given but npw_lr_max is given, eta_lr will be estimated by
        # npw_lr_max (maximium number of PWs allowed).
        # If neither eta_lr nor mesh_lr is given, eta_lr = omega**2 will be used.
        # ERIs of type (cc|dd), (cd|dd), and (dd|dd) will be computed using
        # AFDFT with mesh_lr.
        self.eta_lr = 0.2
        self.cell_fat = None
        self.mesh_lr = None

    def dump_flags(self, verbose=None):
        cell = self.cell
        cell_fat = self.cell_fat
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info('cell num shells = %d, num cGTOs = %d, num pGTOs = %d',
                 cell.nbas, cell.nao_nr(), cell.npgto_nr())
        log.info('omega = %s', self.omega)
        log.info('ke_cutoff = %s', self.ke_cutoff)
        log.info('mesh = %s (%d PWs)', self.mesh, np.prod(self.mesh))
        log.info('mesh_sr = %s (%d PWs)', self.mesh_sr, np.prod(self.mesh_sr))
        if not cell_fat is None:
            log.info('smooth eta_lr = %s', self.eta_lr)
            log.info('mesh_lr = %s (%d PWs)', self.mesh_lr,
                     np.prod(self.mesh_lr))
            log.info('cell_fat num shells = %d, num cGTOs = %d, num pGTOs = %d',
                     cell_fat.nbas, cell_fat.nao_nr(),
                     cell_fat.npgto_nr())
            log.info('         num compact shells = %d, num diffuse shells = %d',
                     *cell_fat._nbas_each_set)
            log.debug('cell-cell_fat bas mapping:%s', "")
            nbas_c = cell_fat._nbas_c
            for ib in range(cell.nbas):
                idx = np.where(cell_fat._bas_idx == ib)[0]
                l = cell.bas_angular(ib)
                if idx.size == 2:
                    log.debug("orig bas %d (l = %d) -> c %d, d %d", ib, l, *idx)
                    log.debug1("  c exp: %s\n  d exp: %s",
                               cell_fat.bas_exp(idx[0]),
                               cell_fat.bas_exp(idx[1]))
                else:
                    btype = "c" if idx[0] < nbas_c else "d"
                    log.debug("orig bas %d (l = %d) -> %s %d", ib, l, btype,
                              idx[0])
                    log.debug1("  %s exp: %s", btype, cell_fat.bas_exp(idx[0]))

        if self.auxcell is None:
            log.info('auxbasis = %s', self.auxbasis)
        else:
            log.info('auxbasis = %s', self.auxcell.basis)

        auxcell = self.auxcell
        if hasattr(auxcell, "_bas_idx"):
            log.info('auxcell num shells = %d, num cGTOs = %d, num pGTOs = %d',
                     auxcell.nbas, auxcell.nao_nr(),
                     auxcell.npgto_nr())
            log.info('        num compact shells = %d, num diffuse shells = %d',
                     *auxcell._nbas_each_set)

        log.info('exp_to_discard = %s', self.exp_to_discard)
        if isinstance(self._cderi, str):
            log.info('_cderi = %s  where DF integrals are loaded (readonly).',
                     self._cderi)
        elif isinstance(self._cderi_to_save, str):
            log.info('_cderi_to_save = %s', self._cderi_to_save)
        else:
            log.info('_cderi_to_save = %s', self._cderi_to_save.name)
        log.info('len(kpts) = %d', len(self.kpts))
        log.debug1('    kpts = %s', self.kpts)
        if self.kpts_band is not None:
            log.info('len(kpts_band) = %d', len(self.kpts_band))
            log.debug1('    kpts_band = %s', self.kpts_band)
        return self

    def gdf_build(self, j_only=None, with_j3c=True, kpts_band=None):
        if self.kpts_band is not None:
            self.kpts_band = np.reshape(self.kpts_band, (-1,3))
        if kpts_band is not None:
            kpts_band = np.reshape(kpts_band, (-1,3))
            if self.kpts_band is None:
                self.kpts_band = kpts_band
            else:
                self.kpts_band = unique(np.vstack((self.kpts_band,kpts_band)))[0]

        self.check_sanity()
        self.dump_flags()

        if self.auxcell is None:
            from pyscf.df.addons import make_auxmol
            self.auxcell = make_auxmol(self.cell, self.auxbasis)

        # Remove duplicated k-points. Duplicated kpts may lead to a buffer
        # located in incore.wrap_int3c larger than necessary. Integral code
        # only fills necessary part of the buffer, leaving some space in the
        # buffer unfilled.
        uniq_idx = unique(self.kpts)[1]
        kpts = np.asarray(self.kpts)[uniq_idx]
        if self.kpts_band is None:
            kband_uniq = np.zeros((0,3))
        else:
            kband_uniq = [k for k in self.kpts_band if len(member(k, kpts))==0]
        if j_only is None:
            j_only = self._j_only
        if j_only:
            kall = np.vstack([kpts,kband_uniq])
            kptij_lst = np.hstack((kall,kall)).reshape(-1,2,3)
        else:
            kptij_lst = [(ki, kpts[j]) for i, ki in enumerate(kpts) for j in range(i+1)]
            kptij_lst.extend([(ki, kj) for ki in kband_uniq for kj in kpts])
            kptij_lst.extend([(ki, ki) for ki in kband_uniq])
            kptij_lst = np.asarray(kptij_lst)

        if with_j3c:
            if isinstance(self._cderi_to_save, str):
                cderi = self._cderi_to_save
            else:
                cderi = self._cderi_to_save.name
            if isinstance(self._cderi, str):
                if self._cderi == cderi and os.path.isfile(cderi):
                    logger.warn(self, 'DF integrals in %s (specified by '
                                '._cderi) is overwritten by GDF '
                                'initialization. ', cderi)
                else:
                    logger.warn(self, 'Value of ._cderi is ignored. '
                                'DF integrals will be saved in file %s .',
                                cderi)
            self._cderi = cderi
            t1 = (time.clock(), time.time())
            self._make_j3c(self.cell, self.auxcell, self.cell_fat, kptij_lst,
                           cderi)
            t1 = logger.timer_debug1(self, 'j3c', *t1)

    def rs_build(self):
        # For each shell, using eta_lr as a cutoff to split it into the diffuse (d) and the compact (c) parts.
        if self.split_basis:
            self.cell_fat = _reorder_cell(self.cell, self.eta_lr)
            if self.cell_fat._nbas_each_set[1] > 0: # has diffuse shells
                self.mesh_lr = _estimate_mesh_lr(self.cell_fat,
                                                 self.cell.precision)
            else:
                self.cell_fat = None    # no split basis happens

        # If omega is not given, estimate an appropriate value for omega
        if self.omega is None:
            # Otherwise, estimate it from the maximum allowed PW size (npw_max)
            omega = estimate_omega_for_npw(self.cell, self.npw_max)
            # The short-range coulomb roughly corresponds to a 1s orb with exponent omega**2. So if omega**2 gets too small, rcut needs to be adjusted.
            # [TODO] adjust rcut based on cell_fat when split basis
            # eomega = omega**2.
            # emin = np.min([np.min(self.cell.bas_exp(ib))
            #               for ib in range(self.cell.nbas)])
            # if eomega < emin:
            #     from pyscf.pbc.gto.cell import _estimate_rcut
            #     rcut = _estimate_rcut(eomega, 0, 1.,
            #                           precision=self.cell.precision)
            #     logger.warn(self, """The squared omega (%.2f) determined from the required PW size (%d PWs) is smaller than the minimum GTO exponents (%.2f). The recommended value for cell.rcut is %s (current value %s)""",
            #                 eomega, self.npw_max, emin, rcut, self.cell.rcut)
            #     raise RuntimeError
            # If basis is split, see if the most diffuse AOs in the compact AO group give a smaller one (hence fewer PWs needed)
            if not self.cell_fat is None:
                nbas_c,nbas_d = self.cell_fat._nbas_each_set
                ec_min = np.min([np.min(self.cell_fat.bas_exp(ib))
                                for ib in range(nbas_c)])
                omega2 = ec_min**-0.5
                if omega2 < omega:
                    omega = omega2
            self.omega = omega

        self.ke_cutoff = df.aft.estimate_ke_cutoff_for_omega(self.cell,
                                                             self.omega)
        self.mesh_sr = pbctools.cutoff_to_mesh(self.cell.lattice_vectors(),
                                               self.ke_cutoff)
        self.mesh_sr = df.df._round_off_to_odd_mesh(self.mesh_sr)

    def build(self, j_only=None, with_j3c=True, kpts_band=None):

        # build for range-separation
        self.rs_build()

        # do normal gdf build with short-range coulomb
        abs_omega = abs(self.omega)
        with self.with_range_coulomb(-abs_omega):
            self.gdf_build(j_only=j_only, with_j3c=with_j3c,
                           kpts_band=kpts_band)

        return self

    def set_range_coulomb(self, omega):
        if omega is None: omega = 0
        self.cell._env[PTR_RANGE_OMEGA] = omega
        if not self.cell_fat is None:
            self.cell_fat._env[PTR_RANGE_OMEGA] = omega

    def with_range_coulomb(self, omega):
        omega0 = self.cell._env[PTR_RANGE_OMEGA].copy()
        return self._TemporaryRSHDFContext(self.set_range_coulomb, (omega,),
                                          (omega0,))

    @contextlib.contextmanager
    def _TemporaryRSHDFContext(self, method, args, args_bak):
        '''Almost every method depends on the Mole environment. Ensure the
        modification in temporary environment being thread safe
        '''
        haslock = hasattr(self, '_lock')
        if not haslock:
            self._lock = threading.RLock()

        with self._lock:
            method(*args)
            try:
                yield
            finally:
                method(*args_bak)
                if not haslock:
                    del self._lock

RSHDF = RangeSeparatedHybridDensityFitting
HDF = RangeSeparatedHybridDensityFitting

def estimate_omega_for_npw(cell, npw_max):
    # bnorm = np.linalg.norm(cell.reciprocal_vectors(), axis=0)
    # d = cell.dimension
    # nm_max = np.floor((npw_max**(1./d)-1) * 0.5)
    # prec = cell.precision
    # omega = (nm_max*bnorm*0.5) / np.log(4*np.pi/((nm_max*bnorm)**2.*prec))**0.5
    # omega = np.min(omega)

    from pyscf.pbc.df.rshdf_helper import _binary_search

    latvecs = cell.lattice_vectors()
    def invomega2meshsize(invomega):
        omega = 1./invomega
        ke_cutoff = df.aft.estimate_ke_cutoff_for_omega(cell, omega)
        mesh = pbctools.cutoff_to_mesh(latvecs, ke_cutoff)
        mesh = df.df._round_off_to_odd_mesh(mesh)
        return np.prod(mesh)

    invomega_rg = 1. / np.asarray([2,0.05])
    invomega, npw = _binary_search(invomega2meshsize, *invomega_rg, npw_max,
                                   0.1, verbose=cell.verbose)
    omega = 1. / invomega

    return omega

def _reorder_cell(cell, eta_smooth, npw_max=None, verbose=None):
    """ Split each shell by eta_smooth into diffuse (d) and compact (c). Then reorder them such that compact shells come first.

    This function is modified from the one under the same name in pyscf/pbc/scf/rsjk.py.
    """
    from pyscf.gto import NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF, ATOM_OF
    log = logger.new_logger(cell, verbose)

    # Split shells based on exponents
    ao_loc = cell.ao_loc_nr()

    cell_fat = copy.copy(cell)

    if not npw_max is None:
        from pyscf.pbc.dft.multigrid import _primitive_gto_cutoff
        # kecuts = _primitive_gto_cutoff(cell, cell.precision)[1]
        meshs = _estimate_mesh_primitive(cell, cell.precision, round2odd=True)

    _env = cell._env.copy()
    compact_bas = []
    diffuse_bas = []
    # xxx_bas_idx maps the shells in the new cell to the original cell
    compact_bas_idx = []
    diffuse_bas_idx = []

    for ib, orig_bas in enumerate(cell._bas):
        nprim = orig_bas[NPRIM_OF]
        nctr = orig_bas[NCTR_OF]

        pexp = orig_bas[PTR_EXP]
        pcoeff = orig_bas[PTR_COEFF]
        es = cell.bas_exp(ib)
        cs = cell._libcint_ctr_coeff(ib)

        if npw_max is None:
            compact_mask = es >= eta_smooth
            diffuse_mask = ~compact_mask
        else:
            npws = np.prod(meshs[ib], axis=1)
            compact_mask = npws > npw_max
            diffuse_mask = ~compact_mask

        c_compact = cs[compact_mask]
        c_diffuse = cs[diffuse_mask]
        _env[pcoeff:pcoeff+nprim*nctr] = np.hstack([
            c_compact.T.ravel(),
            c_diffuse.T.ravel(),
        ])
        _env[pexp:pexp+nprim] = np.hstack([
            es[compact_mask],
            es[diffuse_mask],
        ])

        if c_compact.size > 0:
            bas = orig_bas.copy()
            bas[NPRIM_OF] = c_compact.shape[0]
            bas[PTR_EXP] = pexp
            bas[PTR_COEFF] = pcoeff
            compact_bas.append(bas)
            compact_bas_idx.append(ib)

        if c_diffuse.size > 0:
            bas = orig_bas.copy()
            bas[NPRIM_OF] = c_diffuse.shape[0]
            bas[PTR_EXP] = pexp + c_compact.shape[0]
            bas[PTR_COEFF] = pcoeff + c_compact.size
            diffuse_bas.append(bas)
            diffuse_bas_idx.append(ib)

    cell_fat._env = _env
    cell_fat._bas = np.asarray(compact_bas + diffuse_bas,
                               dtype=np.int32, order='C').reshape(-1, mol_gto.BAS_SLOTS)
    cell_fat._bas_idx = np.asarray(compact_bas_idx + diffuse_bas_idx,
                                   dtype=np.int32)
    cell_fat._nbas_each_set = (len(compact_bas_idx), len(diffuse_bas_idx))
    cell_fat._nbas_c, cell_fat._nbas_d = cell_fat._nbas_each_set

    return cell_fat

def _estimate_mesh_primitive(cell, precision, round2odd=True):
    ''' Estimate the minimum mesh for the diffuse shells.
    '''

    if round2odd:
        fround = lambda x: df.df._round_off_to_odd_mesh(x)
    else:
        fround = lambda x: x

    # from pyscf.pbc.dft.multigrid import _primitive_gto_cutoff
    # kecuts = _primitive_gto_cutoff(cell, cell.precision)[1]

    from pyscf.pbc.gto.cell import _estimate_ke_cutoff
    kecuts = [None] * cell.nbas
    for ib in range(cell.nbas):
        nprim = cell.bas_nprim(ib)
        nctr = cell.bas_nctr(ib)
        es = cell.bas_exp(ib)
        cs = np.max(np.abs(cell.bas_ctr_coeff(ib).reshape(nctr,nprim)), axis=0)
        l = cell.bas_angular(ib)
        kecuts[ib] = _estimate_ke_cutoff(es, l, cs, precision=precision)

    latvecs = cell.lattice_vectors()
    meshs = [None] * cell.nbas
    for ib in range(cell.nbas):
        meshs[ib] = np.asarray([fround(pbctools.cutoff_to_mesh(latvecs, ke))
                               for ke in kecuts[ib]])

    return meshs

def _estimate_mesh_lr(cell_fat, precision):
    ''' Estimate the minimum mesh for the diffuse shells.
    '''
    nc, nd = cell_fat._nbas_each_set
    # from pyscf.pbc.dft.multigrid import _primitive_gto_cutoff
    # kecuts = _primitive_gto_cutoff(cell_fat, cell_fat.precision)[1]
    # kecut = np.max([np.max(kecuts[ib]) for ib in range(nc, cell_fat.nbas)])
    from pyscf.pbc.gto.cell import _estimate_ke_cutoff
    kecut = 0.
    for ib in range(nc,nc+nd):
        nprim = cell_fat.bas_nprim(ib)
        nctr = cell_fat.bas_nctr(ib)
        es = cell_fat.bas_exp(ib)
        cs = np.max(np.abs(cell_fat.bas_ctr_coeff(ib).reshape(nctr,nprim)),
                    axis=0)
        l = cell_fat.bas_angular(ib)
        kecut = max(kecut, np.max(_estimate_ke_cutoff(es, l, cs,
                    precision=precision)))
    mesh_lr = pbctools.cutoff_to_mesh(cell_fat.lattice_vectors(), kecut)

    return mesh_lr

def get_prescreening_data(mydf, cell_fat, extra_precision):
    if mydf.prescreening_type == 1:
        cell_ = mydf.cell if cell_fat is None else cell_fat
        auxcell = mydf.auxcell
        omega = mydf.omega
        Rc_cut, R12_cut_lst = estimate_Rc_R12_cut_max(cell_, auxcell, omega,
                                                      extra_precision)
        return Rc_cut, R12_cut_lst
    elif mydf.prescreening_type == 2:
        cell_ = mydf.cell if cell_fat is None else cell_fat
        auxcell = mydf.auxcell
        omega = mydf.omega
        from pyscf.pbc.df.rshdf_helper import _estimate_Rc_R12_cut as festimate
        Rc_cut_mat, R12_cut_mat = estimate_Rc_R12_cut_SPLIT(cell_, auxcell,
                                                            omega,
                                                            extra_precision,
                                                            festimate)
        return Rc_cut_mat, R12_cut_mat
    elif mydf.prescreening_type == 3:
        cell_ = mydf.cell if cell_fat is None else cell_fat
        auxcell = mydf.auxcell
        omega = mydf.omega
        from pyscf.pbc.df.rshdf_helper import _estimate_Rc_R12_cut2 as festimate
        # Rc_cut_mat, R12_cut_mat = estimate_Rc_R12_cut_SPLIT(cell_, auxcell,
        #                                                     omega,
        #                                                     extra_precision,
        #                                                     festimate)
        Rc_cut_mat, R12_cut_mat = estimate_Rc_R12_cut_SPLIT_batch(
                                        cell_, auxcell, omega, extra_precision)
        return Rc_cut_mat, R12_cut_mat
    else:
        return None


def estimate_Rc_R12_cut_max(cell, auxcell, omega, extra_precision):

    # precision = np.min(extra_precision) * cell.precision
    precision = cell.precision
    print("Searching R12_cut_lst for precision= %.1e" % precision)

    Ls = cell.get_lattice_Ls()
    cell_vol = cell.vol

    eaux = np.min([np.min(auxcell.bas_exp(ib)) for ib in range(auxcell.nbas)])
    ework = np.min([np.min(cell.bas_exp(ib)) for ib in range(cell.nbas)])

    from pyscf.pbc.df.rshdf_helper import _estimate_Rc_R12_cut
    Rc_loc, R12_cut_lst = _estimate_Rc_R12_cut(eaux, ework, ework, omega,
                                               Ls, cell_vol, precision)

    nseg = Rc_loc.size
    print(("%s="+" %4.1f"*nseg) % ("Rc_loc",*Rc_loc))
    print(("%s="+" %4.1f"*nseg) % ("R12_cut_lst",*R12_cut_lst))

    return Rc_loc[-1], R12_cut_lst


def estimate_Rc_R12_cut_split(cell, auxcell, smooth_eta, extra_precision):
    nbas = cell.nbas
    aux_nbas = auxcell.nbas

    echg = smooth_eta
    # for all aux orbs of same angular momentum and eaux > eaux_split, use
    # Rc_cut & R12_cut_lst determined from eaux_split
    # for those smaller than this number, use eaux_min

    Ls = cell.get_lattice_Ls()
    cell_vol = cell.vol

    auxbas_exp = np.asarray([auxcell.bas_exp(ib) for ib in range(aux_nbas)])
    ibaux_compact = np.where(auxbas_exp > echg)[0]
    ibaux_smooth = np.where(auxbas_exp <= echg)[0]
    print(ibaux_compact, np.min(auxbas_exp[ibaux_compact]))
    print(ibaux_smooth, np.min(auxbas_exp[ibaux_smooth]))


def estimate_Rc_R12_cut_SPLIT(cell, auxcell, omega, extra_precision,
                              festimate_Rc_R12_cut):

    nbas = cell.nbas
    aux_nbas = auxcell.nbas

    Ls = cell.get_lattice_Ls()
    cell_vol = cell.vol

    # condense extra_prec from auxcell AO to auxcell shell
    aux_ao_loc = auxcell.ao_loc_nr()
    extra_prec = [np.min(extra_precision[range(*aux_ao_loc[i:i+2])])
                  for i in range(aux_nbas)]

    cell_coords = cell.atom_coords()
    auxcell_coords = auxcell.atom_coords()
    Rc_cut_mat = np.zeros([aux_nbas,nbas,nbas])
    R12_cut_lst = []
    for ibaux in range(aux_nbas):
        eaux = np.min(auxcell.bas_exp(ibaux))
        Raux = auxcell_coords[auxcell.bas_atom(ibaux)]
        for ib in range(nbas):
            ei = np.min(cell.bas_exp(ib))
            Ri = cell_coords[cell.bas_atom(ib)]
            for jb in range(ib+1):
                ej = np.min(cell.bas_exp(jb))
                Rj = cell_coords[cell.bas_atom(jb)]
                R0s = np.vstack([Ri-Raux,Rj-Raux])
                precision = cell.precision * extra_prec[ibaux]
                Rc_loc, R12_cut_lst_ = festimate_Rc_R12_cut(eaux, ei, ej, omega,
                                                            Ls, cell_vol,
                                                            precision, R0s)
                Rc_cut_mat[ibaux,ib,jb] = Rc_cut_mat[ibaux,jb,ib] = Rc_loc[-1]
                R12_cut_lst.append(R12_cut_lst_)

    nc_max = np.max(Rc_cut_mat).astype(int)+1
    R12_cut_mat = np.zeros([aux_nbas,nbas,nbas,nc_max])
    ind = 0
    for ibaux in range(aux_nbas):
        for ib in range(nbas):
            for jb in range(ib+1):
                R12_cut_lst_ = R12_cut_lst[ind]
                nc_ = R12_cut_lst_.size
                R12_cut_mat[ibaux,ib,jb,:nc_] = R12_cut_lst_
                R12_cut_mat[ibaux,ib,jb,nc_:] = R12_cut_lst_[-1]
                R12_cut_mat[ibaux,jb,ib] = R12_cut_mat[ibaux,ib,jb]

                ind += 1

    return Rc_cut_mat, R12_cut_mat


def estimate_Rc_R12_cut_SPLIT_batch(cell, auxcell, omega, extra_precision):

    from pyscf.pbc.df.rshdf_helper import _estimate_Rc_R12_cut2_batch
    aux_ao_loc = auxcell.ao_loc_nr()
    aux_nbas = auxcell.nbas
    extra_prec = [np.min(extra_precision[range(*aux_ao_loc[i:i+2])])
                  for i in range(aux_nbas)]
    auxprecs = np.asarray(extra_prec) * cell.precision
    return _estimate_Rc_R12_cut2_batch(cell, auxcell, omega, auxprecs)
