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
from pyscf.pbc.df import rshdf_helper
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


def get_prescreening_data(mydf, cell_fat, precision=None, extra_precision=None,
                          shlpr_mask=None):
    if mydf.prescreening_type == 0:
        return None

    cell_ = mydf.cell if cell_fat is None else cell_fat
    auxcell = mydf.auxcell
    omega = abs(mydf.omega)
    if mydf.prescreening_type == 3:
        fspltbas = rshdf_helper._estimate_Rc_R12_cut2_batch
    elif mydf.prescreening_type == 4:
        fspltbas = rshdf_helper._estimate_Rc_R12_cut3_batch
    else:
        raise ValueError
    Rc_cut_mat, R12_cut_mat = rshdf_helper.estimate_Rc_R12_cut_SPLIT_batch(
                                                    cell_, auxcell, omega,
                                                    precision, extra_precision,
                                                    fspltbas, shlpr_mask)
    return Rc_cut_mat, R12_cut_mat

def get_safe_rcut(mydf, Rc_cut_mat, R12_cut_mat, mask):
    if not mydf.safe_rcut:
        return mydf.cell.rcut

    lenmax = R12_cut_mat.shape[-1]
    nRcs = Rc_cut_mat[:,mask].astype(int).ravel()
    R12s_lst = R12_cut_mat[:,mask].reshape(-1,lenmax)
    rmax = 0
    for nRc,R12s in zip(nRcs,R12s_lst):
        rmax = max(rmax, R12s[nRc] + nRc)

    def rcut2negrmax(rcut):
        return -np.linalg.norm(mydf.cell.get_lattice_Ls(rcut=rcut),axis=1).max()

    xlo = rmax * 0.5
    xhi = rmax * 0.9
    rprec = mydf.cell.vol**0.3333333333 * 0.5
    rcut, rmax = rshdf_helper._binary_search(rcut2negrmax, xlo, xhi, -rmax,
                                             rprec)
    rmax *= -1

    return rcut

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
    else:
        shlpr_mask_fat_c = shlpr_mask_fat_d = None

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

    nao = cell.nao_nr()
    naux = auxcell.nao_nr()

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)

    log.debug('Num uniq kpts %d', len(uniq_kpts))
    log.debug2('uniq_kpts %s', uniq_kpts)

# compute j2c first as it informs the integral screening in computing j3c
    # short-range part of j2c ~ (-kpt_ji | kpt_ji)
    omega_j2c = abs(mydf.omega_j2c)
    """
    with auxcell.with_range_coulomb(-omega_j2c):
        j2c = auxcell.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)
    """
    j2c_full = auxcell.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)
    with auxcell.with_range_coulomb(omega_j2c):
        j2c_lr = auxcell.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)
    j2c = [j2c_full[k] - j2c_lr[k] for k in range(len(uniq_kpts))]

    # Add (1) short-range G=0 (i.e., charge) part and (2) long-range part
    qaux2 = None
    g0_j2c = np.pi/omega_j2c**2./cell.vol
    mesh_j2c = mydf.mesh_j2c
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh_j2c)
    b = cell.reciprocal_vectors()
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]

    max_memory = max(2000, mydf.max_memory - lib.current_memory()[0])
    blksize = max(2048, int(max_memory*.5e6/16/auxcell.nao_nr()))
    log.debug2('max_memory %s (MB)  blocksize %s', max_memory, blksize)
    for k, kpt in enumerate(uniq_kpts):
        # short-range charge part
        if is_zero(kpt):
            if qaux2 is None:
                qaux2 = np.outer(qaux,qaux)
            j2c[k] -= qaux2 * g0_j2c
        # long-range part via aft
        coulG_lr = weighted_coulG(cell, omega_j2c, kpt, False, mesh_j2c)
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
            if mydf.j2c_eig_always:
                raise scipy.linalg.LinAlgError
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
            ndrop = np.count_nonzero(w<mydf.linear_dep_threshold)
            if ndrop > 0:
                log.debug('DF metric linear dependency for kpt %s',
                          uniq_kptji_id)
                log.debug('cond = %.4g, drop %d bfns', w[-1]/w[0], ndrop)
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
    if mydf.use_extra_precision_R:
        for k in range(len(uniq_kpts)):
            j2c, j2c_negative, j2ctag = cholesky_decomposed_metric(k)
            if j2ctag == "CD":
                j2c_inv = np.eye(j2c.shape[0])
                j2c_inv = scipy.linalg.solve_triangular(j2c, j2c_inv,
                                                        lower=True,
                                                        overwrite_b=True)
            else:
                j2c_inv = j2c
            extra_precision = np.minimum(1./(np.max(np.abs(j2c_inv),
                                         axis=0)+1.), extra_precision)
        log.debug1("Smallest extra precision for j3c latsum: %.3e",
                   np.min(extra_precision))

    prescreening_data = get_prescreening_data(mydf, cell_fat, mydf.precision_R,
                                              extra_precision, shlpr_mask_fat_c)
    if not prescreening_data is None:
        Rc_cut_max = np.max(prescreening_data[0])
        log.debug1("Rc_cut max= %.2f  cell.rcut= %.2f", Rc_cut_max, cell.rcut)
        mask = shl_mask_fat_c > 0 if split_basis else np.ones((cell.nbas,)*2,
                                                              dtype=bool)
        rcut = get_safe_rcut(mydf, *prescreening_data, mask)
    else:
        rcut = cell.rcut
    log.debug1("Use rcut= %.2f to generate Ls for j3c", rcut)
    Ls = cell.get_lattice_Ls(rcut=rcut)

    t1 = log.timer_debug1('prescrn warmup', *t1)

    # short-range part
    if split_auxbasis:
        shls_slice = (0,cell.nbas,0,cell.nbas,0,aux_nbas_c)
    else:
        shls_slice = None

    if mydf.safe_mode:
        # The SR coulomb integrals from libcint can be inaccurate for certain choices of omega (especially larger ones) and/or for certain orbitals (especially higher-angular momentum ones). The safe mode computes SR by full - LR. Note that it requires a larger scratch space (~3x) and runs slower (~2x).
        # This may become innecessary in the future with a better libcint.
        if split_basis:
            rshdf_helper._aux_e2_spltbas(
                            cell, cell_fat, auxcell, fswap, 'int3c2e',
                            aosym='s2', Ls=Ls,
                            kptij_lst=kptij_lst, dataname='j3c-junk_full',
                            max_memory=max_memory,
                            bvk_kmesh=bvk_kmesh,
                            prescreening_type=mydf.prescreening_type,
                            prescreening_data=prescreening_data,
                            shlpr_mask_fat=shlpr_mask_fat_c,
                            shls_slice=shls_slice)
            with mydf.with_range_coulomb(omega):
                rshdf_helper._aux_e2_spltbas(
                                cell, cell_fat, auxcell, fswap, 'int3c2e',
                                aosym='s2', Ls=Ls,
                                kptij_lst=kptij_lst, dataname='j3c-junk_lr',
                                max_memory=max_memory,
                                bvk_kmesh=bvk_kmesh,
                                prescreening_type=mydf.prescreening_type,
                                prescreening_data=prescreening_data,
                                shlpr_mask_fat=shlpr_mask_fat_c,
                                shls_slice=shls_slice)
        else:
            rshdf_helper._aux_e2_nospltbas(
                            cell, auxcell, fswap, 'int3c2e', aosym='s2', Ls=Ls,
                            kptij_lst=kptij_lst, dataname='j3c-junk_full',
                            max_memory=max_memory,
                            bvk_kmesh=bvk_kmesh,
                            prescreening_type=mydf.prescreening_type,
                            prescreening_data=prescreening_data,
                            shls_slice=shls_slice)
            with mydf.with_range_coulomb(omega):
                rshdf_helper._aux_e2_nospltbas(
                                cell, auxcell, fswap, 'int3c2e', aosym='s2', Ls=Ls,
                                kptij_lst=kptij_lst, dataname='j3c-junk_lr',
                                max_memory=max_memory,
                                bvk_kmesh=bvk_kmesh,
                                prescreening_type=mydf.prescreening_type,
                                prescreening_data=prescreening_data,
                                shls_slice=shls_slice)
        # The ordering here mimicks that used by _aux_e2_(no)spltbas
        sorted_ij_idx = np.hstack([np.where(uniq_inverse == k)[0]
                                  for k, kpt in enumerate(uniq_kpts)])
        nseg = len(fswap['j3c-junk_lr/0'])
        for iseg in range(nseg):
            for ij in sorted_ij_idx:
                psr = 'j3c-junk/%d/%d'%(ij,iseg)
                plr = 'j3c-junk_lr/%d/%d'%(ij,iseg)
                pfull = 'j3c-junk_full/%d/%d'%(ij,iseg)
                fswap[psr] = fswap[pfull][()] - fswap[plr][()]
                del fswap[pfull]
                del fswap[plr]
    else:
        with mydf.with_range_coulomb(-omega):
            if split_basis:
                rshdf_helper._aux_e2_spltbas(
                                cell, cell_fat, auxcell, fswap, 'int3c2e',
                                aosym='s2', Ls=Ls,
                                kptij_lst=kptij_lst, dataname='j3c-junk',
                                max_memory=max_memory,
                                bvk_kmesh=bvk_kmesh,
                                prescreening_type=mydf.prescreening_type,
                                prescreening_data=prescreening_data,
                                shlpr_mask_fat=shlpr_mask_fat_c,
                                shls_slice=shls_slice)
            else:
                rshdf_helper._aux_e2_nospltbas(
                                cell, auxcell, fswap, 'int3c2e', aosym='s2', Ls=Ls,
                                kptij_lst=kptij_lst, dataname='j3c-junk',
                                max_memory=max_memory,
                                bvk_kmesh=bvk_kmesh,
                                prescreening_type=mydf.prescreening_type,
                                prescreening_data=prescreening_data,
                                shls_slice=shls_slice)
    Ls = None
    t1 = log.timer_debug1('3c2e', *t1)

    prescreening_data = None

    # recompute g0 and Gvectors for j3c
    g0 = np.pi/omega**2./cell.vol
    mesh = mydf.mesh_compact
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]
    if split_basis:
        coords = cell.gen_uniform_grids(mesh)

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
                    for iap_fat, iap in rshdf_helper.fat_orig_loop(
                                                cell_fat, cell, aosym='s1',
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
        buflen = min(max(int(max_memory*.38e6/16/naux/(nkptj+1)), 1),
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
        def pw_contract(istep, sh_range, j3cR, j3cI):
            bstart, bend, ncol = sh_range
            if aosym == 's2':
                shls_slice = (bstart, bend, 0, bend)
            else:
                shls_slice = (bstart, bend, 0, cell.nbas)

            if split_basis:
                astart = aopr_loc[aosym][bstart]
                aend = aopr_loc[aosym][bend]
                mask_c = aopr_mask_c[aosym][astart:aend]
                mask_d = aopr_mask_d[aosym][astart:aend]
                has_c = mask_c.any()
                has_d = mask_d.any()

                # long-range coulomb for cc and cd
                if has_c:
                    for p0, p1 in lib.prange(0, ngrids, Gblksize):
                        nG = p1 - p0
                        tick_ = np.asarray([time.clock(), time.time()])
                        dat = rshdf_helper.ft_aopair_kpts_spltbas(
                                                cell_fat, cell, Gv[p0:p1],
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
                    # Unlike AFT, FFT can't batch G. We instead batch kptj here.
                    kblksize = min(nkptj, int(np.floor(
                                   buf.size / float(ncol * ngrids))))
                    for k0,k1 in lib.prange(0, nkptj, kblksize):
                        log.debug1("kjseg: %d-%d/%d", k0, k1, nkptj)
                        tick_ = np.asarray([time.clock(), time.time()])
                        # dat = ft_aopair_kpts_spltbas(cell_fat, cell, Gv[p0:p1],
                        #                              shls_slice, aosym,
                        #                              b, gxyz[p0:p1], Gvbase,
                        #                              kpt, adapted_kptjs,
                        #                              out=buf,
                        #                              bvk_kmesh=bvk_kmesh,
                        #                              shlpr_mask=shlpr_mask_fat_d)
                        dat = rshdf_helper.fft_aopair_kpts_spltbas(
                                                mydf._numint, cell_fat, cell,
                                                mesh, coords, aosym=aosym,
                                                q=kpt,
                                                kptjs=adapted_kptjs[k0:k1],
                                                shls_slice0=shls_slice,
                                                shl_mask=shl_mask_fat_d,
                                                out=buf)
                        tock_ = np.asarray([time.clock(), time.time()])
                        tspans[0] += tock_ - tick_

                        for ik,k in enumerate(range(k0,k1)):
                            aoao = dat[ik].reshape(ngrids,ncol)
                            for p0, p1 in lib.prange(0, ngrids, Gblksize):
                                nG = p1 - p0
                                pqkR = np.ndarray((ncol,nG), buffer=pqkRbuf)
                                pqkI = np.ndarray((ncol,nG), buffer=pqkIbuf)
                                pqkR[:] = aoao[p0:p1].real.T
                                pqkI[:] = aoao[p0:p1].imag.T

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


class RangeSeparatedHybridDensityFitting2(df.df.GDF):
    '''Range Separated Hybrid Density Fitting
    '''

    # class methods defined outside the class
    _make_j3c = _make_j3c

    def __init__(self, cell, kpts=np.zeros((1,3))):
        df.df.GDF.__init__(self, cell, kpts=kpts)

        self.use_bvkcell = False    # set True to use bvkcell for j3c & AFT
        self.prescreening_type = 3
        self.split_basis = False
        self.split_auxbasis = False
        self.safe_mode = False  # if True, SR = Full - LR is used for j3c

        self.kpts = kpts
        self.kpts_band = None
        self._auxbasis = None

        # precision for real-space lattice sum (R) and reciprocal-space Fourier transform (G).
        # Both are are set to cell.precision by default and will be modified by the extra_precision determined from inverting j2c (see _make_j3c).
        self.precision_R = self.cell.precision
        self.precision_G = self.cell.precision
        # extra_precision_G allows extra precision in determining the diffuse AOs that are treated by the PW basis of size <= npw_max. Numerical tests on several simple solids (C/SiC/MgO/LiF) suggest that 1e-4 is a good choice: it stabilizes the calculation when cell.precision is low (e.g., >= 1e-8), while having virtually no effects when cell.precision is high (i.e., does not lower the performance).
        self.extra_precision_G = 1e-2

        # One of {omega, npw_max} must be provided, and the other will be deduced automatically from it. The priority when both are given is omega > npw_max.
        # The default is npw_max = 13^3 for Gamma point and 7^3 otherwise, which has been tested to be a good choice balancing accuracy and speed.
        # Once omega is determined, mesh_compact is determined for (L|g^lr|pq) to achieve given accuracy, where L = C and pq = cc/cd.
        self.npw_max = 2250 if is_zero(kpts) else 350
        self.omega = None
        self.ke_cutoff = None
        self.mesh_compact = None

        # omega and mesh for j2c. Since j2c is to be inverted, it is desirable to use (1) a "fixed" omega for reproducibility, and (2) a higher precision to minimize the error caused by inversion.
        # This extra computational cost due to the higher precision is negligible compared to the j3c build.
        # FOR EXPERTS: if you want omega_j2c to be the same as omega, set omega_j2c to be any negative number.
        self.omega_j2c = 0.4
        self.mesh_j2c = None
        self.precision_j2c = 1e-8 * self.precision_G

        # set True to force calculating j2c^(-1/2) using eigenvalue decomposition (ED); otherwise, Cholesky decomposition (CD) is used first, and ED is called only if CD fails.
        self.j2c_eig_always = False

        # If split_basis is True, each ao shell will be split into a diffuse (d) part and a compact (c) part based on the pGTO exponents & coeffs.
        # The criterion is such that a "d" shell must be expressed by a PW basis of size self.mesh_compact to achieve self.precision_G.
        # (C|cc), (C|cd) will be computed using range-separation
        # (D|cc), (D|cd), (C|dd), (D|dd) will be computed in G-space completely.
        self.cell_fat = None

        # if AO basis is split, numint is needed for FFT evaluating (*|dd)
        self._numint = None

        # For debugging and should be removed later
        self.round2odd = False  # if True, mesh for j3c will be rounded to odd
        self.use_extra_precision_R = True
        self.safe_rcut = True

    def dump_flags(self, verbose=None):
        cell = self.cell
        cell_fat = self.cell_fat
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info('cell num shells = %d, num cGTOs = %d, num pGTOs = %d',
                 cell.nbas, cell.nao_nr(), cell.npgto_nr())
        log.info('safe_mode = %s', self.safe_mode)
        log.info('precision_R = %s', self.precision_R)
        log.info('precision_G = %s', self.precision_G)
        log.info('extra_precision_G = %s', self.extra_precision_G)
        log.info('j2c_eig_always = %s', self.j2c_eig_always)
        log.info('omega = %s', self.omega)
        log.info('ke_cutoff = %s', self.ke_cutoff)
        log.info('mesh = %s (%d PWs)', self.mesh, np.prod(self.mesh))
        log.info('mesh_compact = %s (%d PWs)', self.mesh_compact,
                 np.prod(self.mesh_compact))
        if not cell_fat is None:
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
                    log.debug2("  c cff: %s\n  d cff: %s",
                               cell_fat._libcint_ctr_coeff(idx[0]),
                               cell_fat._libcint_ctr_coeff(idx[1]))
                else:
                    btype = "c" if idx[0] < nbas_c else "d"
                    log.debug("orig bas %d (l = %d) -> %s %d", ib, l, btype,
                              idx[0])
                    log.debug1("  %s exp: %s", btype, cell_fat.bas_exp(idx[0]))
                    log.debug2("  %s cff: %s", btype,
                               cell_fat._libcint_ctr_coeff(idx[0]))

        if self.auxcell is None:
            log.info('auxbasis = %s', self.auxbasis)
        else:
            log.info('auxbasis = %s', self.auxcell.basis)
            log.info('auxcell precision= %s', self.auxcell.precision)
            log.info('auxcell rcut = %s', self.auxcell.rcut)
            log.info('omega_j2c = %s', self.omega_j2c)
            log.info('mesh_j2c = %s (%d PWs)', self.mesh_j2c,
                     np.prod(self.mesh_j2c))

        auxcell = self.auxcell
        log.info('auxcell num shells = %d, num cGTOs = %d, num pGTOs = %d',
                 auxcell.nbas, auxcell.nao_nr(),
                 auxcell.npgto_nr())
        if hasattr(auxcell, "_bas_idx"):
            log.info('        num compact shells = %d, num diffuse shells = %d',
                     *auxcell._nbas_each_set)
            log.debug1('diffuse auxshls:%s', '')
            for ib in range(auxcell._nbas_c,auxcell.nbas):
                log.debug1('shlidx= %d, l= %d, exp= %.5g, coeff= %.5g',
                           ib, auxcell.bas_angular(ib), auxcell.bas_exp(ib),
                           auxcell.bas_ctr_coeff(ib))

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

        # for debugging and should be removed later
        log.info('\ndebugging flags%s', '')
        log.info('round2odd for j3c = %s', self.round2odd)
        log.info('use_extra_precision_R = %s', self.use_extra_precision_R)
        log.info('safe_rcut = %s', self.safe_rcut)
        log.info('%s', '')

        return self

    def _rsh_build(self):
        # find kmax
        kpts = self.kpts if self.kpts_band is None else np.vstack([self.kpts,
                                                                self.kpts_band])
        b = self.cell.reciprocal_vectors()
        scaled_kpts = np.linalg.solve(b.T, kpts.T).T
        scaled_kpts[scaled_kpts > 0.49999999] -= 1
        kpts = np.dot(scaled_kpts, b)
        kmax = np.linalg.norm(kpts, axis=-1).max()

        # If omega is not given, estimate it from npw_max
        r2o = self.round2odd
        if self.omega is None:
            self.omega, self.ke_cutoff, self.mesh_compact = \
                                rshdf_helper.estimate_omega_for_npw(
                                                self.cell, self.npw_max,
                                                self.precision_G,
                                                kmax=kmax,
                                                round2odd=r2o)
        else:
            self.ke_cutoff, self.mesh_compact = \
                                rshdf_helper.estimate_mesh_for_omega(
                                                self.cell, self.omega,
                                                self.precision_G,
                                                kmax=kmax,
                                                round2odd=r2o)

        # For each shell, using npw_max to split into c and d parts such that d shells can be well-described by a PW of size self.mesh_compact
        precision_fat = self.precision_G * self.extra_precision_G
        if self.split_basis:
            self.cell_fat = rshdf_helper._reorder_cell(self.cell, 0,
                                                       self.npw_max,
                                                       precision_fat,
                                                       round2odd=r2o)
            if self.cell_fat._nbas_each_set[1] > 0: # has diffuse shells
                from pyscf.pbc.dft import numint
                self._numint = numint.KNumInt()
            else:
                self.cell_fat = None    # no split basis happens

        # As explained in __init__, if negative omega_j2c --> use omega
        if self.omega_j2c < 0: self.omega_j2c = self.omega

        # build auxcell and split its basis if requested
        # Note 1) higher precision is used fo determining mesh_j2c as explained in __init__, but for basis split, the same precision_fat is used as above.
        # Note 2) round2odd is required for j2c to conserve conjugation symmetry
        # Note 3) unlike AOs, auxiliary basis is all primitive, so _reorder_cell won't split any shells -- just reorder them so that compact shells come first. Thus, there's no need to differentiate auxcell and auxcell_fat.
        from pyscf.df.addons import make_auxmol
        auxcell = make_auxmol(self.cell, self.auxbasis)
        auxcell.precision = self.precision_j2c
        auxcell.rcut = max([auxcell.bas_rcut(ib, auxcell.precision)
                            for ib in range(auxcell.nbas)])
        self.mesh_j2c = rshdf_helper.estimate_mesh_for_omega(
                                auxcell, self.omega_j2c, round2odd=True)[1]

        if self.split_auxbasis:
            auxcell_fat = rshdf_helper._reorder_cell(auxcell, 0, self.npw_max,
                                                     precision_fat,
                                                     round2odd=r2o)
            if auxcell_fat._nbas_each_set[1] > 0: # has diffuse shells
                auxcell = auxcell_fat

        self.auxcell = auxcell

    def _kpts_build(self, kpts_band=None):
        if self.kpts_band is not None:
            self.kpts_band = np.reshape(self.kpts_band, (-1,3))
        if kpts_band is not None:
            kpts_band = np.reshape(kpts_band, (-1,3))
            if self.kpts_band is None:
                self.kpts_band = kpts_band
            else:
                self.kpts_band = unique(np.vstack((self.kpts_band,kpts_band)))[0]

    def _gdf_build(self, j_only=None, with_j3c=True):
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

    def build(self, j_only=None, with_j3c=True, kpts_band=None):
        # formatting k-points
        self._kpts_build(kpts_band=kpts_band)

        # build for range-separation hybrid
        self._rsh_build()

        # dump flags before the final build
        self.check_sanity()
        self.dump_flags()

        # do normal gdf build with the modified _make_j3c
        self._gdf_build(j_only=j_only, with_j3c=with_j3c)

        return self

    def set_range_coulomb(self, omega):
        self.cell._env[PTR_RANGE_OMEGA] = omega
        self.auxcell._env[PTR_RANGE_OMEGA] = omega
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

RSHDF2 = RangeSeparatedHybridDensityFitting2
HDF2 = RangeSeparatedHybridDensityFitting2
