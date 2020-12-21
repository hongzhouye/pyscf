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
Hybrid density fitting 2

The HDF class in rshdf.py does not reproduce GDF results because the fitting is done using the short-range Coulomb interaction as the metric. The HDF2 class fixes this issue. The basic idea is to split the j2c and j3c integrals as follows
j2c:
        C    D
    C (C|C) (C|D)
    D (D|C) (D|D)

j3c:
       cc      cd     dd
    C (C|cc) (C|cd) (C|dd)
    D (D|cc) (D|cd) (D|dd)

Then the short-range part of
    j2c: (C|C),
    j3c: (C|cc), (C|cd)
are evaluated using real-space lattice sum, while their long-range part as well as the full Coulomb integral of
    j2c: (C|D), (D|C), (D|D).
    j3c: (C|dd), (D|cc), (D|cd), (D|dd)
are evaluated in the reciprocal space.

All the techniques derived for the original HDF class (integral screening, split basis, etc.) can be used here straightforwardly.
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
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import outcore
from pyscf.pbc.df import rshdf_ao2mo
from pyscf.pbc.df.rshdf_helper import (ft_aopair_kpts_spltbas,
                                       fft_aopair_kpts_spltbas, fat_orig_loop)
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


def weighted_coulG(cell, omega, kpt=np.zeros(3), exx=False, mesh=None):
    if cell.omega != 0:
        raise RuntimeError('RangeSeparatedHybridDensityFitting2 cannot be used '
                           'to evaluate the long-range HF exchange in RSH '
                           'functional')
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    if abs(omega) < 1.e-10:
        omega_ = None
    else:
        omega_ = omega
    coulG = pbctools.get_coulG(cell, kpt, False, None, mesh, Gv,
                               omega=omega_)
    coulG *= kws
    return coulG


def get_shlpr_aopr_mask(cell, cell_fat):
    n_compact, n_diffuse = cell_fat._nbas_each_set
    bas_idx = cell_fat._bas_idx
    nbas_fat = cell_fat.nbas
    shlpr_mask_fat_c = np.ones((nbas_fat, nbas_fat), dtype=np.int8,
                               order="C")
    shlpr_mask_fat_c[n_compact:,n_compact:] = 0
    shlpr_mask_fat_d = 1 - shlpr_mask_fat_c

    return shlpr_mask_fat_c, shlpr_mask_fat_d


def get_aopr_mask(cell, cell_fat, shlpr_mask_fat_c, shlpr_mask_fat_d):
    nbas = cell.nbas
    nao = cell.nao
    ao_loc = cell.ao_loc_nr()
    bas_idx = cell_fat._bas_idx
    mask_mat_c = np.ones((nao,nao), dtype=bool)
    mask_mat_d = np.ones((nao,nao), dtype=bool)
    fatbas_by_orig = [np.where(bas_idx==ib)[0] for ib in range(nbas)]
    for ib in range(nbas):
        ibs_fat = fatbas_by_orig[ib]
        i0,i1 = ao_loc[ib:ib+2]
        for jb in range(nbas):
            jbs_fat = fatbas_by_orig[jb]
            j0,j1 = ao_loc[jb:jb+2]
            mask_mat_c[i0:i1,j0:j1] = (shlpr_mask_fat_c[
                                       np.ix_(ibs_fat,jbs_fat)]).any()
            mask_mat_d[i0:i1,j0:j1] = (shlpr_mask_fat_d[
                                       np.ix_(ibs_fat,jbs_fat)]).any()

    tril_idx = np.tril_indices_from(mask_mat_c)
    # aosym = 's2' and 's1', respectively
    aopr_mask_c = {"s2": mask_mat_c[tril_idx], "s1": np.ravel(mask_mat_c)}
    aopr_mask_d = {"s2": mask_mat_d[tril_idx], "s1": np.ravel(mask_mat_d)}

    ao_loc = cell.ao_loc_nr()
    aopr_loc = {"s2": ao_loc*(ao_loc+1)//2, "s1": ao_loc*cell.nao_nr()}

    return aopr_mask_c, aopr_mask_d, aopr_loc


# kpti == kptj: s2 symmetry
# kpti == kptj == 0 (gamma point): real
def _make_j3c(mydf, cell, auxcell, cell_fat, kptij_lst, cderi_file):
    t1 = (time.clock(), time.time())
    log = logger.Logger(mydf.stdout, mydf.verbose)
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])

    omega = abs(mydf.omega)

    if mydf.use_bvkcell:
        bvk_kmesh = k2gamma.kpts_to_kmesh(cell, mydf.kpts)
    else:
        bvk_kmesh = None

    if hasattr(auxcell, "_nbas_c"):
        split_auxbasis = True
        aux_nbas_c, aux_nbas_d = auxcell._nbas_each_set
        aux_ao_loc = auxcell.ao_loc_nr()
        aux_nao_c = aux_ao_loc[aux_nbas_c]
        aux_nao = aux_ao_loc[-1]
        aux_nao_d = aux_nao - aux_nao_c
    else:
        split_auxbasis = False

    split_basis = not cell_fat is None
    if split_basis:
        shl_mask_fat_c = np.ones(cell_fat.nbas, dtype=bool)
        shl_mask_fat_c[cell_fat._nbas_c:] = 0
        shl_mask_fat_d = ~shl_mask_fat_c
        shlpr_mask_fat_c, shlpr_mask_fat_d = get_shlpr_aopr_mask(cell, cell_fat)
        aopr_mask_c, aopr_mask_d, aopr_loc = get_aopr_mask(cell, cell_fat,
                                                           shlpr_mask_fat_c,
                                                           shlpr_mask_fat_d)

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
    qaux = df.rshdf.get_aux_chg(auxcell)
    g0 = np.pi/mydf.omega**2./cell.vol

    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    mesh = mydf.mesh_sr
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    b = cell.reciprocal_vectors()
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]
    if split_basis:
        coords = cell.gen_uniform_grids(mesh)

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)

    log.debug('Num uniq kpts %d', len(uniq_kpts))
    log.debug2('uniq_kpts %s', uniq_kpts)

# compute j2c first as it informs the integral screening in computing j3c
    # short-range part of j2c ~ (-kpt_ji | kpt_ji)
    """
    with auxcell.with_range_coulomb(-omega):
        j2c = auxcell.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)
    """
    j2c_full = auxcell.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)
    with auxcell.with_range_coulomb(omega):
        j2c_lr = auxcell.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)
    j2c = [j2c_full[k] - j2c_lr[k] for k in range(len(uniq_kpts))]

    # Add (1) short-range G=0 (i.e., charge) part and (2) long-range part
    qaux2 = None
    max_memory = max(2000, mydf.max_memory - lib.current_memory()[0])
    blksize = max(2048, int(max_memory*.5e6/16/auxcell.nao_nr()))
    log.debug2('max_memory %s (MB)  blocksize %s', max_memory, blksize)
    for k, kpt in enumerate(uniq_kpts):
        # short-range charge part
        if is_zero(kpt):
            if qaux2 is None:
                qaux2 = np.outer(qaux,qaux)
            j2c[k] -= qaux2 * g0
        # long-range part via aft
        coulG_lr = weighted_coulG(cell, omega, kpt, False, mesh)
        for p0, p1 in lib.prange(0, ngrids, blksize):
            aoaux = ft_ao.ft_ao(auxcell, Gv[p0:p1], None, b, gxyz[p0:p1],
                                Gvbase, kpt).T
            LkR = np.asarray(aoaux.real, order='C')
            LkI = np.asarray(aoaux.imag, order='C')
            aoaux = None

            if is_zero(kpt):  # kpti == kptj
                j2c[k] += lib.ddot(LkR*coulG_lr[p0:p1], LkR.T)
                j2c[k] += lib.ddot(LkI*coulG_lr[p0:p1], LkI.T)
            else:
                j2cR, j2cI = df.df_jk.zdotCN(LkR*coulG_lr[p0:p1],
                                             LkI*coulG_lr[p0:p1], LkR.T, LkI.T)
                j2c[k] += j2cR + j2cI * 1j
            LkR = LkI = None
        fswap['j2c/%d'%k] = j2c[k]
    j2c = coulG_lr = None

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
    # inverting j2c, and use it's column max to determine an extra precision for 3c2e prescreening
    extra_precision = np.ones(auxcell.nao_nr())
    for k in range(len(uniq_kpts)):
        j2c, j2c_negative, j2ctag = cholesky_decomposed_metric(k)
        if j2ctag == "CD":
            j2c_inv = np.eye(j2c.shape[0])
            j2c_inv = scipy.linalg.solve_triangular(j2c, j2c_inv, lower=True,
                                                    overwrite_b=True)
        else:
            j2c_inv = j2c
        extra_precision = np.minimum(1./(np.max(np.abs(j2c_inv), axis=0)+1.),
                                      extra_precision)

    from pyscf.pbc.df.rshdf import get_prescreening_data
    prescreening_data = get_prescreening_data(mydf, cell_fat, extra_precision)

    t1 = log.timer_debug1('prescrn warmup', *t1)

    # short-range part
    if split_auxbasis:
        shls_slice = (0,cell.nbas,0,cell.nbas,0,aux_nbas_c)
    else:
        shls_slice = None

    if split_basis:
        with cell.with_range_coulomb(-omega), \
             auxcell.with_range_coulomb(-omega), \
             cell_fat.with_range_coulomb(-omega):
            outcore._aux_e2_hy_spltbas(
                               cell, cell_fat, auxcell, fswap, 'int3c2e', aosym='s2',
                               kptij_lst=kptij_lst, dataname='j3c-junk',
                               max_memory=max_memory,
                               bvk_kmesh=bvk_kmesh,
                               prescreening_type=mydf.prescreening_type,
                               prescreening_data=prescreening_data,
                               shlpr_mask_fat=shlpr_mask_fat_c,
                               shls_slice=shls_slice)
    else:
        with cell.with_range_coulomb(-omega), \
             auxcell.with_range_coulomb(-omega):
            outcore._aux_e2_hy(cell, auxcell, fswap, 'int3c2e', aosym='s2',
                               kptij_lst=kptij_lst, dataname='j3c-junk',
                               max_memory=max_memory,
                               bvk_kmesh=bvk_kmesh,
                               prescreening_type=mydf.prescreening_type,
                               prescreening_data=prescreening_data,
                               shls_slice=shls_slice)
    t1 = log.timer_debug1('3c2e', *t1)

    prescreening_data = None

    # mute charges for diffuse auxiliary shells
    if split_auxbasis:
        qaux = qaux[:aux_nao_c]

    # Add (1) short-range G=0 (i.e., charge) part and (2) long-range part
    tspans = np.zeros((5,2))    # ft_aop, pw_cntr, j2c_cntr, write, read
    tspannames = ["ft_aop", "pw_cntr", "j2c_cntr", "write", "read"]
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

        shls_slice = (0, auxcell.nbas)
        Gaux = ft_ao.ft_ao(auxcell, Gv, shls_slice, b, gxyz, Gvbase, kpt)
        wcoulG_lr = weighted_coulG(cell, omega, kpt, False, mesh)
        if split_basis or split_auxbasis:
            wcoulG = weighted_coulG(cell, 0, kpt, False, mesh)
        if split_basis:
            Gaux_d = Gaux * wcoulG.reshape(-1,1)
            kLR_d = Gaux_d.real.copy('C')
            kLI_d = Gaux_d.imag.copy('C')
            Gaux_d = None
        if split_auxbasis:
            Gaux[:,:aux_nao_c] *= wcoulG_lr.reshape(-1,1)
            Gaux[:,aux_nao_c:] *= wcoulG.reshape(-1,1)
        else:
            Gaux *= wcoulG_lr.reshape(-1,1)
        kLR = Gaux.real.copy('C')
        kLI = Gaux.imag.copy('C')
        Gaux = None

        if is_zero(kpt):  # kpti == kptj
            aosym = 's2'
            nao_pair = nao*(nao+1)//2

            if cell.dimension == 3:
                vbar = qaux * g0
                if split_basis: # only compute ovlp for cc and cd
                    nao_fat = cell_fat.nao_nr()
                    ovlp_fat = cell_fat.pbc_intor('int1e_ovlp', hermi=1,
                                                  kpts=adapted_kptjs)
                    ovlp_fat = [s.ravel() for s in ovlp_fat]
                    nkj = len(ovlp_fat)
                    ovlp = [np.zeros((nao*nao), dtype=ovlp_fat[k].dtype)
                            for k in range(nkj)]
                    for iap_fat, iap in fat_orig_loop(cell_fat, cell,
                                                      aosym='s1',
                                                      shlpr_mask=shlpr_mask_fat_c):
                        for k in range(nkj):
                            ovlp[k][iap] += ovlp_fat[k][iap_fat]
                    ovlp = [lib.pack_tril(s.reshape(nao,nao)) for s in ovlp]
                else:
                    ovlp = cell.pbc_intor('int1e_ovlp', hermi=1,
                                          kpts=adapted_kptjs)
                    ovlp = [lib.pack_tril(s) for s in ovlp]
        else:
            aosym = 's1'
            nao_pair = nao**2

        mem_now = lib.current_memory()[0]
        log.debug2('memory = %s', mem_now)
        max_memory = max(2000, mydf.max_memory-mem_now)
        # nkptj for 3c-coulomb arrays plus 1 Lpq array
        nbuf_extra = 1 if split_basis and mydf.split_basis_pwcntr == 1 else 0
        buflen = min(max(int(max_memory*.38e6/16/naux/(nkptj+1+nbuf_extra)), 1),
                     nao_pair)
        shranges = _guess_shell_ranges(cell, buflen, aosym)
        buflen = max([x[2] for x in shranges])
        # +1 for a pqkbuf
        if aosym == 's2':
            Gblksize = max(16, int(max_memory*.1e6/16/buflen/(nkptj+1)))
        else:
            Gblksize = max(16, int(max_memory*.2e6/16/buflen/(nkptj+1)))
        Gblksize = min(Gblksize, ngrids, 16384)
        pqkRbuf = np.empty(buflen*Gblksize)
        pqkIbuf = np.empty(buflen*Gblksize)
        # buf for ft_aopair
        buf = np.empty(nkptj*buflen*Gblksize, dtype=np.complex128)
        if split_basis and mydf.split_basis_pwcntr == 1:
            ncol_max = np.max([shrg[-1] for shrg in shranges])
            jbuf = np.empty(ncol_max*naux, dtype=np.float64)
        def pw_contract(istep, sh_range, j3cR, j3cI):
            bstart, bend, ncol = sh_range
            if aosym == 's2':
                shls_slice = (bstart, bend, 0, bend)
            else:
                shls_slice = (bstart, bend, 0, cell.nbas)

            if split_basis and mydf.split_basis_pwcntr == 1:
                astart = aopr_loc[aosym][bstart]
                aend = aopr_loc[aosym][bend]
                mask_c = aopr_mask_c[aosym][astart:aend]
                mask_d = aopr_mask_d[aosym][astart:aend]
                ncol_c = np.sum(mask_c)
                ncol_d = np.sum(mask_d)
                has_c = mask_c.any()
                has_d = mask_d.any()

                for p0, p1 in lib.prange(0, ngrids, Gblksize):
                    nG = p1 - p0
                    if has_c:
                        tick_ = np.asarray([time.clock(), time.time()])
                        # long-range coulomb for cc and cd
                        dat = ft_aopair_kpts_spltbas(cell_fat, cell, Gv[p0:p1],
                                                     shls_slice, aosym,
                                                     b, gxyz[p0:p1], Gvbase,
                                                     kpt, adapted_kptjs,
                                                     out=buf,
                                                     bvk_kmesh=bvk_kmesh,
                                                     shlpr_mask=shlpr_mask_fat_c)
                        tock_ = np.asarray([time.clock(), time.time()])
                        tspans[0] += tock_ - tick_
                        for k, ji in enumerate(adapted_ji_idx):
                            aoao = dat[k].reshape(nG,ncol)[:,mask_c]
                            pqkR = np.ndarray((ncol_c,nG), buffer=pqkRbuf)
                            pqkI = np.ndarray((ncol_c,nG), buffer=pqkIbuf)
                            pqkR[:] = aoao.real.T
                            pqkI[:] = aoao.imag.T

                            j3c_ = np.ndarray((naux,ncol_c), buffer=jbuf)
                            lib.dot(kLR[p0:p1].T, pqkR.T, 1, j3c_, 0)
                            lib.dot(kLI[p0:p1].T, pqkI.T, 1, j3c_, 1)
                            j3cR[k][:,mask_c] += j3c_
                            if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                                lib.dot(kLR[p0:p1].T, pqkI.T, 1, j3c_, 0)
                                lib.dot(kLI[p0:p1].T, pqkR.T, -1, j3c_, 1)
                                j3cI[k][:,mask_c] += j3c_
                        tick_ = np.asarray([time.clock(), time.time()])
                        tspans[1] += tick_ - tock_

                    # add full coulomb for dd
                    if has_d:
                        tick_ = np.asarray([time.clock(), time.time()])
                        dat = ft_aopair_kpts_spltbas(cell_fat, cell, Gv[p0:p1],
                                                     shls_slice, aosym,
                                                     b, gxyz[p0:p1], Gvbase,
                                                     kpt, adapted_kptjs,
                                                     out=buf,
                                                     bvk_kmesh=bvk_kmesh,
                                                     shlpr_mask=shlpr_mask_fat_d)
                        tock_ = np.asarray([time.clock(), time.time()])
                        tspans[0] += tock_ - tick_

                        for k, ji in enumerate(adapted_ji_idx):
                            aoao = dat[k].reshape(nG,ncol)[:,mask_d]
                            pqkR = np.ndarray((ncol_d,nG), buffer=pqkRbuf)
                            pqkI = np.ndarray((ncol_d,nG), buffer=pqkIbuf)
                            pqkR[:] = aoao.real.T
                            pqkI[:] = aoao.imag.T

                            j3c_ = np.ndarray((naux,ncol_d), buffer=jbuf)
                            lib.dot(kLR_d[p0:p1].T, pqkR.T, 1, j3c_, 0)
                            lib.dot(kLI_d[p0:p1].T, pqkI.T, 1, j3c_, 1)
                            j3cR[k][:,mask_d] += j3c_
                            if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                                lib.dot(kLR_d[p0:p1].T, pqkI.T, 1, j3c_, 0)
                                lib.dot(kLI_d[p0:p1].T, pqkR.T, -1, j3c_, 1)
                                j3cI[k][:,mask_d] += j3c_
                        tick_ = np.asarray([time.clock(), time.time()])
                        tspans[1] += tick_ - tock_
            elif split_basis and mydf.split_basis_pwcntr == 2:
                astart = aopr_loc[aosym][bstart]
                aend = aopr_loc[aosym][bend]
                mask_c = aopr_mask_c[aosym][astart:aend]
                mask_d = aopr_mask_d[aosym][astart:aend]
                has_c = mask_c.any()
                has_d = mask_d.any()

                for p0, p1 in lib.prange(0, ngrids, Gblksize):
                    nG = p1 - p0
                    if has_c:
                        tick_ = np.asarray([time.clock(), time.time()])
                        # long-range coulomb for cc and cd
                        dat = ft_aopair_kpts_spltbas(cell_fat, cell, Gv[p0:p1],
                                                     shls_slice, aosym,
                                                     b, gxyz[p0:p1], Gvbase,
                                                     kpt, adapted_kptjs,
                                                     out=buf,
                                                     bvk_kmesh=bvk_kmesh,
                                                     shlpr_mask=shlpr_mask_fat_c)
                        tock_ = np.asarray([time.clock(), time.time()])
                        tspans[0] += tock_ - tick_
                        for k, ji in enumerate(adapted_ji_idx):
                            aoao = dat[k].reshape(nG,ncol)
                            pqkR = np.ndarray((ncol,nG), buffer=pqkRbuf)
                            pqkI = np.ndarray((ncol,nG), buffer=pqkIbuf)
                            pqkR[:] = aoao.real.T
                            pqkI[:] = aoao.imag.T

                            lib.dot(kLR[p0:p1].T, pqkR.T, 1, j3cR[k], 1)
                            lib.dot(kLI[p0:p1].T, pqkI.T, 1, j3cR[k], 1)
                            if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                                lib.dot(kLR[p0:p1].T, pqkI.T, 1, j3cI[k], 1)
                                lib.dot(kLI[p0:p1].T, pqkR.T, -1, j3cI[k], 1)
                        tick_ = np.asarray([time.clock(), time.time()])
                        tspans[1] += tick_ - tock_

                    # add full coulomb for dd
                    if has_d:
                        tick_ = np.asarray([time.clock(), time.time()])
                        # dat = ft_aopair_kpts_spltbas(cell_fat, cell, Gv[p0:p1],
                        #                              shls_slice, aosym,
                        #                              b, gxyz[p0:p1], Gvbase,
                        #                              kpt, adapted_kptjs,
                        #                              out=buf,
                        #                              bvk_kmesh=bvk_kmesh,
                        #                              shlpr_mask=shlpr_mask_fat_d)
                        dat = fft_aopair_kpts_spltbas(mydf._numint,
                                                      cell_fat, cell,
                                                      mesh, coords,
                                                      aosym=aosym, q=kpt,
                                                      kptjs=adapted_kptjs,
                                                      shls_slice0=shls_slice,
                                                      shl_mask=shl_mask_fat_d,
                                                      out=buf)
                        tock_ = np.asarray([time.clock(), time.time()])
                        tspans[0] += tock_ - tick_

                        for k, ji in enumerate(adapted_ji_idx):
                            aoao = dat[k].reshape(nG,ncol)
                            pqkR = np.ndarray((ncol,nG), buffer=pqkRbuf)
                            pqkI = np.ndarray((ncol,nG), buffer=pqkIbuf)
                            pqkR[:] = aoao.real.T
                            pqkI[:] = aoao.imag.T

                            lib.dot(kLR_d[p0:p1].T, pqkR.T, 1, j3cR[k], 1)
                            lib.dot(kLI_d[p0:p1].T, pqkI.T, 1, j3cR[k], 1)
                            if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                                lib.dot(kLR_d[p0:p1].T, pqkI.T, 1, j3cI[k], 1)
                                lib.dot(kLI_d[p0:p1].T, pqkR.T, -1, j3cI[k], 1)
                        tick_ = np.asarray([time.clock(), time.time()])
                        tspans[1] += tick_ - tock_
            else:
                for p0, p1 in lib.prange(0, ngrids, Gblksize):
                    tick_ = np.asarray([time.clock(), time.time()])
                    dat = ft_ao.ft_aopair_kpts(cell, Gv[p0:p1], shls_slice,
                                               aosym, b, gxyz[p0:p1], Gvbase,
                                               kpt, adapted_kptjs, out=buf,
                                               bvk_kmesh=bvk_kmesh)
                    tock_ = np.asarray([time.clock(), time.time()])
                    tspans[0] += tock_ - tick_
                    nG = p1 - p0
                    for k, ji in enumerate(adapted_ji_idx):
                        aoao = dat[k].reshape(nG,ncol)
                        pqkR = np.ndarray((ncol,nG), buffer=pqkRbuf)
                        pqkI = np.ndarray((ncol,nG), buffer=pqkIbuf)
                        pqkR[:] = aoao.real.T
                        pqkI[:] = aoao.imag.T

                        lib.dot(kLR[p0:p1].T, pqkR.T, 1, j3cR[k][:], 1)
                        lib.dot(kLI[p0:p1].T, pqkI.T, 1, j3cR[k][:], 1)
                        if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                            lib.dot(kLR[p0:p1].T, pqkI.T, 1, j3cI[k][:], 1)
                            lib.dot(kLI[p0:p1].T, pqkR.T, -1, j3cI[k][:], 1)
                    tick_ = np.asarray([time.clock(), time.time()])
                    tspans[1] += tick_ - tock_

            for k, ji in enumerate(adapted_ji_idx):
                tick_ = np.asarray([time.clock(), time.time()])
                if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                    v = j3cR[k]
                else:
                    v = j3cR[k] + j3cI[k] * 1j
                if j2ctag == 'CD':
                    v = scipy.linalg.solve_triangular(j2c, v, lower=True, overwrite_b=True)
                else:
                    v = lib.dot(j2c, v)
                tock_ = np.asarray([time.clock(), time.time()])
                tspans[2] += tock_ - tick_
                feri['j3c/%d/%d'%(ji,istep)] = v
                tick_ = np.asarray([time.clock(), time.time()])
                tspans[3] += tick_ - tock_

                # low-dimension systems
                if j2c_negative is not None:
                    feri['j3c-/%d/%d'%(ji,istep)] = lib.dot(j2c_negative, v)

        with lib.call_in_background(pw_contract) as compute:
            col1 = 0
            for istep, sh_range in enumerate(shranges):
                log.debug1('int3c2e [%d/%d], AO [%d:%d], ncol = %d',
                           istep+1, len(shranges), *sh_range)
                bstart, bend, ncol = sh_range
                col0, col1 = col1, col1+ncol
                j3cR = []
                j3cI = []
                tick_ = np.asarray([time.clock(), time.time()])
                for k, idx in enumerate(adapted_ji_idx):
                    v = np.vstack([fswap['j3c-junk/%d/%d'%(idx,i)][0,col0:col1].T
                                      for i in range(nsegs)])
                    if split_auxbasis:
                        v = np.vstack([v, np.zeros((aux_nao_d,col1-col0),
                                      dtype=v.dtype)])
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
                tock_ = np.asarray([time.clock(), time.time()])
                tspans[4] += tock_ - tick_
                compute(istep, sh_range, j3cR, j3cI)
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

    # report time for aft part
    for tspan, tspanname in zip(tspans, tspannames):
        log.debug1("    CPU time for %s %9.2f sec, wall time %9.2f sec",
                   "%10s"%tspanname, *tspan)
    log.debug1("%s", "")


class RangeSeparatedHybridDensityFitting2(df.rshdf.RSHDF):
    '''Range Separated Hybrid Density Fitting
    '''

    # class methods defined outside the class
    _make_j3c = _make_j3c

    # ao2mo
    get_eri = get_ao_eri = rshdf_ao2mo.get_eri_sr
    # ao2mo = get_mo_eri = rshdf_ao2mo.general
    # ao2mo_7d = rshdf_ao2mo.ao2mo_7d

    def __init__(self, cell, kpts=np.zeros((1,3))):
        df.rshdf.RSHDF.__init__(self, cell, kpts=kpts)

        self.npw_max = 350
        self.split_basis_pwcntr = 1 # 1 or 2

    def rs2_build(self):
        """ make auxcell
        """
        from pyscf.df.addons import make_auxmol
        auxcell = make_auxmol(self.cell, self.auxbasis)

        # Reorder auxiliary basis such that compact shells come first.
        # Note that unlike AOs, auxiliary basis is all primitive, so _reorder_cell won't split any shells -- just reorder them. So there's no need to differentiate auxcell and auxcell_fat.
        if self.split_auxbasis:
            from pyscf.pbc.df.rshdf import _reorder_cell, _estimate_mesh_lr
            auxcell_fat = _reorder_cell(auxcell, self.eta, self.npw_max)
            if auxcell_fat._nbas_each_set[1] > 0: # has diffuse shells
                auxcell = auxcell_fat

        self.auxcell = auxcell

        # self.cell_fat = None    # for now

    def build(self, j_only=None, with_j3c=True, kpts_band=None):
        # build for range-separation
        self.rs_build()
        self.rs2_build()

        # do normal gdf build (NOTE: unlike in RSHDF.build(), here we do not globally use short-range coulomb, and let _make_j3c to decide which part is computed with short-range and which with long-range.)
        self.gdf_build(j_only=j_only, with_j3c=with_j3c, kpts_band=kpts_band)

        return self

RSHDF2 = RangeSeparatedHybridDensityFitting2
HDF2 = RangeSeparatedHybridDensityFitting2
