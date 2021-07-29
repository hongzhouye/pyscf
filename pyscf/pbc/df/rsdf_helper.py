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

import sys
import ctypes
import copy
import h5py
import time
import numpy as np
import scipy.special

from pyscf import gto as mol_gto
from pyscf.pbc import df
from pyscf.pbc.df.intor_j2c import binary_search
from pyscf.pbc.df.intor_j2c import get_2c2e_Rcut, get_atom_Rcuts_2c
from pyscf.pbc.df.intor_j3c import (get_refuniq_map, get_schwartz_data,
                                    get_schwartz_dcut, make_dijs_lst,
                                    get_3c2e_Rcuts, get_atom_Rcuts_3c,
                                    get_Lsmin, get_bvk_data)
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique, KPT_DIFF_TOL
from pyscf.pbc import tools as pbctools
from pyscf.scf import _vhf
from pyscf.pbc.gto import _pbcintor
from pyscf.pbc.tools import k2gamma
from pyscf import lib
from pyscf.lib import logger
from pyscf.lib.parameters import BOHR

libpbc = lib.load_library('libpbc')


""" Helper functions for determining omega/mesh and basis splitting
"""
def estimate_ke_cutoff_for_omega_kpt_corrected(cell, omega, precision, kmax):
    fac = 32*np.pi**2    # Qiming
    # fac = 4 * cell.vol / np.pi  # Hongzhou
    ke_cutoff = -2*omega**2 * np.log(precision / (fac*omega**2))
    ke_cutoff = ((2*ke_cutoff)**0.5 + kmax)**2. * 0.5
    return ke_cutoff

def estimate_omega_for_npw(cell, npw_max, precision=None, kmax=0,
                           round2odd=True):

    if precision is None: precision = cell.precision
    # TODO: add extra precision for small omega ~ 2*omega / np.pi**0.5
    latvecs = cell.lattice_vectors()

    def omega2all(omega):
        ke_cutoff = estimate_ke_cutoff_for_omega_kpt_corrected(cell, omega,
                                                               precision, kmax)
        mesh = pbctools.cutoff_to_mesh(latvecs, ke_cutoff)
        if round2odd:
            mesh = df.df._round_off_to_odd_mesh(mesh)
        return ke_cutoff, mesh
    def fcheck(omega):
        return np.prod(omega2all(omega)[1]) > npw_max
    omega_rg = np.asarray([0.05,2])
    omega = binary_search(*omega_rg, 0.02, False, fcheck)
    ke_cutoff, mesh = omega2all(omega)

    return omega, ke_cutoff, mesh

def estimate_mesh_for_omega(cell, omega, precision=None, kmax=0,
                            round2odd=True):

    if precision is None: precision = cell.precision
    ke_cutoff = estimate_ke_cutoff_for_omega_kpt_corrected(cell, omega,
                                                           precision, kmax)
    mesh = pbctools.cutoff_to_mesh(cell.lattice_vectors(), ke_cutoff)
    if round2odd:
        mesh = df.df._round_off_to_odd_mesh(mesh)

    return ke_cutoff, mesh

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
        cs = np.max(np.abs(cell.bas_ctr_coeff(ib)), axis=1)
        l = cell.bas_angular(ib)
        kecuts[ib] = _estimate_ke_cutoff(es, l, cs, precision=precision)

    latvecs = cell.lattice_vectors()
    meshs = [None] * cell.nbas
    for ib in range(cell.nbas):
        meshs[ib] = np.asarray([fround(pbctools.cutoff_to_mesh(latvecs, ke))
                               for ke in kecuts[ib]])

    return meshs

def _estimate_mesh_lr(cell_fat, precision, round2odd=True):
    ''' Estimate the minimum mesh for the diffuse shells.
    '''
    if round2odd:
        fround = lambda x: df.df._round_off_to_odd_mesh(x)
    else:
        fround = lambda x: x

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
        cs = np.max(np.abs(cell_fat.bas_ctr_coeff(ib)), axis=1)
        l = cell_fat.bas_angular(ib)
        kecut = max(kecut, np.max(_estimate_ke_cutoff(es, l, cs,
                    precision=precision)))
    mesh_lr = fround(pbctools.cutoff_to_mesh(cell_fat.lattice_vectors(), kecut))

    return mesh_lr

def _reorder_cell(cell, eta_smooth, npw_max=None, precision=None,
                  round2odd=True, verbose=None):
    """ Split each shell by eta_smooth or npw_max into diffuse (d) and compact (c). Then reorder them such that compact shells come first.

    This function is modified from the one under the same name in pyscf/pbc/scf/rsjk.py.
    """
    if precision is None: precision = cell.precision

    from pyscf.gto import NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF, ATOM_OF
    log = logger.new_logger(cell, verbose)

    # Split shells based on exponents
    ao_loc = cell.ao_loc_nr()

    cell_fat = copy.copy(cell)

    if not npw_max is None:
        from pyscf.pbc.dft.multigrid import _primitive_gto_cutoff
        meshs = _estimate_mesh_primitive(cell, precision, round2odd=round2odd)
        eta_safe = 10.

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
            # The "_estimate_ke_cutoff" function sometimes fails for pGTOs with very large exponents. Here we enforce all pGTOs whose exponents are greater than eta_safe to be compact.
            compact_mask = (npws > npw_max) | (es > eta_safe)
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

""" short-range j2c via screened lattice sum
"""
def intor_j2c(cell, omega, precision=None, kpts=None, hermi=1, shls_slice=None,
# +++++++ Use the default for the following unless you know what you are doing
              lmp=True, lasympt=True,
              eta_correct=True, R_correct=False, vol_correct=False,
# -------
# +++++++ debug options
              no_screening=False,   # set Rcuts to effectively infinity
# -------
            ):
    log = logger.Logger(cell.stdout, cell.verbose)

    t1 = np.asarray([logger.process_clock(), logger.perf_counter()])

    intor = "int2c2e"
    intor, comp = mol_gto.moleintor._get_intor_and_comp(
                                            cell._add_suffix(intor), None)
    assert(comp == 1)

# prescreening data
    if precision is None: precision = cell.precision

    refuniqshl_map, uniq_atms, uniq_bas, uniq_bas_loc = get_refuniq_map(cell)
    Rcuts = get_2c2e_Rcut(uniq_bas, cell.vol, omega, precision,
                          lmp=lmp, lasympt=lasympt,
                          eta_correct=eta_correct, R_correct=R_correct,
                          vol_correct=vol_correct)
    Rcut2s = np.ones(Rcuts)*1e20 if no_screening else Rcuts**2.
    atom_Rcuts = get_atom_Rcuts_2c(Rcuts, uniq_bas_loc)
    cell_rcut = atom_Rcuts.max()
    Ls = get_Lsmin(cell, atom_Rcuts, uniq_atms)
    log.debug1("j2c prescreening: cell rcut %.2f Bohr  keep %d imgs",
               cell_rcut, Ls.shape[0])
    t1 = log.timer_debug1('prescrn warmup', *t1)
# end prescreening data

    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    if hermi == 0:
        aosym = 's1'
    else:
        aosym = 's2'
    fill = getattr(libpbc, 'PBCnr_2c2e_fill_k'+aosym)
    fintor = getattr(mol_gto.moleintor.libcgto, intor)
    cintopt = lib.c_null_ptr()

    pcell = copy.copy(cell)
    pcell.precision = min(cell.precision, cell.precision)
    pcell._atm, pcell._bas, pcell._env = \
            atm, bas, env = mol_gto.conc_env(cell._atm, cell._bas, cell._env,
                                             cell._atm, cell._bas, cell._env)
    env[mol_gto.PTR_RANGE_OMEGA] = -abs(omega)
    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas)
    i0, i1, j0, j1 = shls_slice[:4]
    j0 += cell.nbas
    j1 += cell.nbas
    ao_loc = mol_gto.moleintor.make_loc(bas, intor)
    ni = ao_loc[i1] - ao_loc[i0]
    nj = ao_loc[j1] - ao_loc[j0]

    out = np.empty((nkpts,comp,ni,nj), dtype=np.complex128)

    expkL = np.asarray(np.exp(1j*np.dot(kpts_lst, Ls.T)), order='C')
    drv = libpbc.PBCnr_2c2e_k_drv

    drv(fintor, fill, out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nkpts), ctypes.c_int(comp), ctypes.c_int(len(Ls)),
        Ls.ctypes.data_as(ctypes.c_void_p),
        expkL.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int*4)(i0, i1, j0, j1),
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
        refuniqshl_map.ctypes.data_as(ctypes.c_void_p),
        Rcut2s.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(pcell.natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(pcell.nbas),
        env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size))

    mat = []
    for k, kpt in enumerate(kpts_lst):
        v = out[k]
        if hermi != 0:
            for ic in range(comp):
                lib.hermi_triu(v[ic], hermi=hermi, inplace=True)
        if comp == 1:
            v = v[0]
        if abs(kpt).sum() < 1e-9:  # gamma_point
            v = v.real
        mat.append(v)

    if kpts is None or np.shape(kpts) == (3,):  # A single k-point
        mat = mat[0]

    t1 = log.timer_debug1('j2c latsum', *t1)

    return mat

""" Helper functions for short-range j3c via real space lattice sum
    Modified from pyscf.pbc.df.outcore/incore
"""
def _aux_e2_nospltbas(cell, auxcell_or_auxbasis, omega, erifile,
                      intor='int3c2e',
                      aosym='s2ij', Ls=None, comp=None, kptij_lst=None,
                      dataname='eri_mo', shls_slice=None, max_memory=2000,
                      bvk_kmesh=None,
                      precision=None,
                      fac_type="ME",
                      eta_correct=True, R_correct=True,
                      vol_correct_d=False, vol_correct_R=False,
                      dstep=1,  # unit: Angstrom
                      verbose=0):
    r'''3-center AO integrals (ij|L) with double lattice sum:
    \sum_{lm} (i[l]j[m]|L[0]), where L is the auxiliary basis.
    Three-index integral tensor (kptij_idx, nao_pair, naux) or four-index
    integral tensor (kptij_idx, comp, nao_pair, naux) are stored on disk.

    **This function should be only used by RSGDF initialization function
    _make_j3c**

    Args:
        kptij_lst : (*,2,3) array
            A list of (kpti, kptj)
    '''
    log = logger.Logger(cell.stdout, cell.verbose)

    if isinstance(auxcell_or_auxbasis, mol_gto.Mole):
        auxcell = auxcell_or_auxbasis
    else:
        auxcell = make_auxcell(cell, auxcell_or_auxbasis)

# prescreening data
    t1 = (logger.process_clock(), logger.perf_counter())
    if precision is None: precision = cell.precision
    refuniqshl_map, uniq_atms, uniq_bas, uniq_bas_loc = get_refuniq_map(cell)
    auxuniqshl_map, uniq_atms, uniq_basaux, uniq_basaux_loc = \
                                                        get_refuniq_map(auxcell)

    dstep_BOHR = dstep / BOHR
    Qauxs = get_schwartz_data(uniq_basaux, omega, keep1ctr=False, safe=True)
    dcuts = get_schwartz_dcut(uniq_bas, cell.vol, omega, precision/Qauxs.max(),
                              r0=cell.rcut, vol_correct=vol_correct_d)
    dijs_lst = make_dijs_lst(dcuts, dstep_BOHR)
    dijs_loc = np.cumsum([0]+[len(dijs) for dijs in dijs_lst]).astype(np.int32)
    if fac_type.upper() in ["ISFQ0","ISFQL"]:
        Qs_lst = get_schwartz_data(uniq_bas, omega, dijs_lst, keep1ctr=True,
                                   safe=True)
    else:
        Qs_lst = [np.zeros_like(dijs) for dijs in dijs_lst]
    Rcuts = get_3c2e_Rcuts(uniq_bas, uniq_basaux, dijs_lst, cell.vol, omega,
                           precision, fac_type, Qs_lst,
                           eta_correct=eta_correct, R_correct=R_correct,
                           vol_correct=vol_correct_R)
    bas_exps = np.array([np.asarray(b[1:])[:,0].min() for b in uniq_bas])
    atom_Rcuts = get_atom_Rcuts_3c(Rcuts, dijs_lst, bas_exps, uniq_bas_loc,
                                   uniq_basaux_loc)
    cell_rcut = atom_Rcuts.max()
    uniqexp = np.array([np.asarray(b[1:])[:,0].min() for b in uniq_bas])
    nbasauxuniq = len(uniq_basaux)
    dcut2s = dcuts**2.
    Rcut2s = Rcuts**2.
    Ls = get_Lsmin(cell, atom_Rcuts, uniq_atms)
    prescreening_data = (refuniqshl_map, auxuniqshl_map, nbasauxuniq, uniqexp,
                         dcut2s, dstep_BOHR, Rcut2s, dijs_loc, Ls)
    log.debug("j3c prescreening: cell rcut %.2f Bohr  keep %d imgs",
              cell_rcut, Ls.shape[0])
    t1 = log.timer_debug1('prescrn warmup', *t1)
# prescreening data ends here

    intor, comp = mol_gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)

    if isinstance(erifile, h5py.Group):
        feri = erifile
    elif h5py.is_hdf5(erifile):
        feri = h5py.File(erifile, 'a')
    else:
        feri = h5py.File(erifile, 'w')
    if dataname in feri:
        del(feri[dataname])
    if dataname+'-kptij' in feri:
        del(feri[dataname+'-kptij'])

    if kptij_lst is None:
        kptij_lst = np.zeros((1,2,3))
    feri[dataname+'-kptij'] = kptij_lst

    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)

    shlpr_mask = np.ones((shls_slice[1]-shls_slice[0],
                          shls_slice[3]-shls_slice[2]),
                          dtype=np.int8, order="C")

    ao_loc = cell.ao_loc_nr()
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)[:shls_slice[5]+1]
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    nkptij = len(kptij_lst)

    nii = (ao_loc[shls_slice[1]]*(ao_loc[shls_slice[1]]+1)//2 -
           ao_loc[shls_slice[0]]*(ao_loc[shls_slice[0]]+1)//2)
    nij = ni * nj

    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]
    aosym_ks2 = abs(kpti-kptj).sum(axis=1) < KPT_DIFF_TOL
    j_only = np.all(aosym_ks2)
    #aosym_ks2 &= (aosym[:2] == 's2' and shls_slice[:2] == shls_slice[2:4])
    aosym_ks2 &= aosym[:2] == 's2'

    if j_only and aosym[:2] == 's2':
        assert(shls_slice[2] == 0)
        nao_pair = nii
    else:
        nao_pair = nij

    if gamma_point(kptij_lst):
        dtype = np.double
    else:
        dtype = np.complex128

    buflen = max(8, int(max_memory*.47e6/16/(nkptij*ni*nj*comp)))
    auxdims = aux_loc[shls_slice[4]+1:shls_slice[5]+1] - aux_loc[shls_slice[4]:shls_slice[5]]
    from pyscf.ao2mo.outcore import balance_segs
    auxranges = balance_segs(auxdims, buflen)
    buflen = max([x[2] for x in auxranges])
    buf = np.empty(nkptij*comp*ni*nj*buflen, dtype=dtype)
    buf1 = np.empty_like(buf)
    bufmem = buf.size*16/1024**2.
    if bufmem > max_memory * 0.5:
        raise RuntimeError("Computing 3c2e integrals requires %.2f MB memory, which exceeds the given maximum memory %.2f MB. Try giving PySCF more memory." % (bufmem*2., max_memory))

    int3c = wrap_int3c_nospltbas(cell, auxcell, shlpr_mask, prescreening_data,
                                 intor, aosym, comp, kptij_lst,
                                 bvk_kmesh=bvk_kmesh)

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
# sorted_ij_idx: Sort and group the kptij_lst according to the ordering in
# df._make_j3c to reduce the data fragment in the hdf5 file.  When datasets
# are written to hdf5, they are saved sequentially. If the integral data are
# saved as the order of kptij_lst, removing the datasets in df._make_j3c will
# lead to holes that can not be reused.
    sorted_ij_idx = np.hstack([np.where(uniq_inverse == k)[0]
                                  for k, kpt in enumerate(uniq_kpts)])

    tril_idx = np.tril_indices(ni)
    tril_idx = tril_idx[0] * ni + tril_idx[1]
    def save(istep, mat):
        tspan = np.zeros((2))
        t1 = np.zeros((2))
        t2 = np.zeros((2))
        for k in sorted_ij_idx:
            v = mat[k]
            if gamma_point(kptij_lst[k]):
                v = v.real
            if aosym_ks2[k] and nao_pair == ni**2:
                v = v[:,tril_idx]
            t1[:] = [logger.process_clock(), logger.perf_counter()]
            feri['%s/%d/%d' % (dataname,k,istep)] = v
            t2[:] = [logger.process_clock(), logger.perf_counter()]
            tspan += t2 - t1
        log.debug1("    CPU time for %s %9.2f sec, wall time %9.2f sec",
                   "     save %d"%istep, *tspan)

    with lib.call_in_background(save) as bsave:
        for istep, auxrange in enumerate(auxranges):
            sh0, sh1, nrow = auxrange
            sub_slice = (shls_slice[0], shls_slice[1],
                         shls_slice[2], shls_slice[3],
                         shls_slice[4]+sh0, shls_slice[4]+sh1)
            mat = np.ndarray((nkptij,comp,nao_pair,nrow), dtype=dtype, buffer=buf)
            t1 = (logger.process_clock(), logger.perf_counter())
            bsave(istep, int3c(sub_slice, mat))
            t1 = log.timer_debug1('cmpt+save %d'%istep, *t1)
            buf, buf1 = buf1, buf

    if not isinstance(erifile, h5py.Group):
        feri.close()
    return erifile

def wrap_int3c_nospltbas(cell, auxcell, shlpr_mask, prescreening_data,
                         intor='int3c2e', aosym='s1',
                         comp=1, kptij_lst=np.zeros((1,2,3)),
                         cintopt=None, bvk_kmesh=None):
    log = logger.Logger(cell.stdout, cell.verbose)

    refuniqshl_map, auxuniqshl_map, nbasauxuniq, uniqexp, dcut2s, dstep_BOHR, Rcut2s, dijs_loc, Ls = prescreening_data

# GTO data
    intor = cell._add_suffix(intor)
    pcell = copy.copy(cell)
    pcell._atm, pcell._bas, pcell._env = \
    atm, bas, env = mol_gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    ao_loc = mol_gto.moleintor.make_loc(bas, intor)
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)
    ao_loc = np.asarray(np.hstack([ao_loc, ao_loc[-1]+aux_loc[1:]]),
                           dtype=np.int32)
    atm, bas, env = mol_gto.conc_env(atm, bas, env,
                                 auxcell._atm, auxcell._bas, auxcell._env)
    nimgs = len(Ls)
    nbas = cell.nbas

    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]

    if bvk_kmesh is None:
        Ls_ = Ls
    else:
        Ls, Ls_, cell_loc_bvk = get_bvk_data(cell, Ls, bvk_kmesh)
        bvk_nimgs = Ls_.shape[0]

    if gamma_point(kptij_lst):
        assert(aosym[:2] == "s2")
        kk_type = 'g'
        dtype = np.double
        nkpts = nkptij = 1
        kptij_idx = np.array([0], dtype=np.int32)
        expkL = np.ones(1)
    elif is_zero(kpti-kptj):  # j_only
        kk_type = 'k'
        dtype = np.complex128
        kpts = kptij_idx = np.asarray(kpti, order='C')
        expkL = np.exp(1j * np.dot(kpts, Ls_.T))
        nkpts = nkptij = len(kpts)
    else:
        kk_type = 'kk'
        dtype = np.complex128
        kpts = unique(np.vstack([kpti,kptj]))[0]
        expkL = np.exp(1j * np.dot(kpts, Ls_.T))
        wherei = np.where(abs(kpti.reshape(-1,1,3)-kpts).sum(axis=2)
                          < KPT_DIFF_TOL)[1]
        wherej = np.where(abs(kptj.reshape(-1,1,3)-kpts).sum(axis=2)
                          < KPT_DIFF_TOL)[1]
        nkpts = len(kpts)
        kptij_idx = np.asarray(wherei*nkpts+wherej, dtype=np.int32)
        nkptij = len(kptij_lst)

    if cintopt is None:
        if nbas > 0:
            cintopt = _vhf.make_cintopt(atm, bas, env, intor)
        else:
            cintopt = lib.c_null_ptr()
# Remove the precomputed pair data because the pair data corresponds to the
# integral of cell #0 while the lattice sum moves shls to all repeated images.
        if intor[:3] != 'ECP':
            libpbc.CINTdel_pairdata_optimizer(cintopt)

    cfunc_prefix = "PBCnr3c"
    if not (gamma_point(kptij_lst) or bvk_kmesh is None):
        cfunc_prefix += "_bvk"
    fill = "%s_%s%s" % (cfunc_prefix, kk_type, aosym[:2])
    drv = getattr(libpbc, "%s_%s_drv"%(cfunc_prefix,kk_type))

    log.debug("Using %s to evaluate SR integrals", fill)

    if gamma_point(kptij_lst):
        def int3c(shls_slice, out):
            shls_slice = (shls_slice[0], shls_slice[1],
                          nbas+shls_slice[2], nbas+shls_slice[3],
                          nbas*2+shls_slice[4], nbas*2+shls_slice[5])
            drv(getattr(libpbc, intor), getattr(libpbc, fill),
                out.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(comp), ctypes.c_int(nimgs),
                Ls.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int*6)(*shls_slice),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                cintopt,
                shlpr_mask.ctypes.data_as(ctypes.c_void_p),  # shlpr_mask
                refuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                auxuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbasauxuniq),
                uniqexp.ctypes.data_as(ctypes.c_void_p),
                dcut2s.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_double(dstep_BOHR),
                Rcut2s.ctypes.data_as(ctypes.c_void_p),
                dijs_loc.ctypes.data_as(ctypes.c_void_p),
                atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
                bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbas),  # need to pass cell.nbas to libpbc.PBCnr3c_drv
                env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size)
                )
            return out

    elif is_zero(kpti-kptj):  # j_only

        raise NotImplementedError

    else:

        if bvk_kmesh is None:
            raise NotImplementedError
        else:
            def int3c(shls_slice, out):
                shls_slice = (shls_slice[0], shls_slice[1],
                              nbas+shls_slice[2], nbas+shls_slice[3],
                              nbas*2+shls_slice[4], nbas*2+shls_slice[5])
                drv(getattr(libpbc, intor), getattr(libpbc, fill),
                    out.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nkptij), ctypes.c_int(nkpts),
                    ctypes.c_int(comp), ctypes.c_int(nimgs),
                    ctypes.c_int(bvk_nimgs),
                    Ls.ctypes.data_as(ctypes.c_void_p),
                    expkL.ctypes.data_as(ctypes.c_void_p),
                    kptij_idx.ctypes.data_as(ctypes.c_void_p),
                    (ctypes.c_int*6)(*shls_slice),
                    ao_loc.ctypes.data_as(ctypes.c_void_p),
                    cintopt,
                    cell_loc_bvk.ctypes.data_as(ctypes.c_void_p),   # cell_loc_bvk
                    shlpr_mask.ctypes.data_as(ctypes.c_void_p),  # shlpr_mask
                    refuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                    auxuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nbasauxuniq),
                    uniqexp.ctypes.data_as(ctypes.c_void_p),
                    dcut2s.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_double(dstep_BOHR),
                    Rcut2s.ctypes.data_as(ctypes.c_void_p),
                    dijs_loc.ctypes.data_as(ctypes.c_void_p),
                    atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
                    bas.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nbas),  # need to pass cell.nbas to libpbc.PBCnr3c_drv
                    env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size)
                    )
                return out

    return int3c

def _aux_e2_spltbas(cell, cell_fat, auxcell_or_auxbasis, omega, erifile,
                    intor='int3c2e', aosym='s2ij',
                    comp=None, kptij_lst=None,
                    dataname='eri_mo', shls_slice=None, max_memory=2000,
                    bvk_kmesh=None, shlpr_mask_fat=None,
                    precision=None,
                    fac_type="ME",
                    eta_correct=True, R_correct=True,
                    vol_correct_d=False, vol_correct_R=False,
                    dstep=1,  # unit: Angstrom
                    verbose=0):
    r'''3-center AO integrals (ij|L) with double lattice sum:
    \sum_{lm} (i[l]j[m]|L[0]), where L is the auxiliary basis.
    Three-index integral tensor (kptij_idx, nao_pair, naux) or four-index
    integral tensor (kptij_idx, comp, nao_pair, naux) are stored on disk.

    **This function should be only used by df and mdf initialization function
    _make_j3c**

    Args:
        kptij_lst : (*,2,3) array
            A list of (kpti, kptj)
    '''
    log = logger.Logger(cell.stdout, cell.verbose)

    if isinstance(auxcell_or_auxbasis, mol_gto.Mole):
        auxcell = auxcell_or_auxbasis
    else:
        auxcell = make_auxcell(cell, auxcell_or_auxbasis)

# prescreening data
    cell_ = cell_fat
    t1 = (logger.process_clock(), logger.perf_counter())
    if precision is None: precision = cell_.precision
    refuniqshl_map, uniq_atms, uniq_bas, uniq_bas_loc = get_refuniq_map(cell_)
    auxuniqshl_map, uniq_atms, uniq_basaux, uniq_basaux_loc = \
                                                        get_refuniq_map(auxcell)

    dstep_BOHR = dstep / BOHR
    Qauxs = get_schwartz_data(uniq_basaux, omega, keep1ctr=False, safe=True)
    dcuts = get_schwartz_dcut(uniq_bas, cell_.vol, omega, precision/Qauxs.max(),
                              r0=cell_.rcut, vol_correct=vol_correct_d)
    dijs_lst = make_dijs_lst(dcuts, dstep_BOHR)
    nbas = cell_.nbas
    nbas2 = nbas*(nbas+1)//2
    for i in range(nbas):
        for j in range(i+1):
            ij = i*(i+1)//2 + j
            print(i,j, dcuts[ij])
    dijs_loc = np.cumsum([0]+[len(dijs) for dijs in dijs_lst]).astype(np.int32)
    if fac_type.upper() in ["ISFQ0","ISFQL"]:
        Qs_lst = get_schwartz_data(uniq_bas, omega, dijs_lst, keep1ctr=True,
                                   safe=True)
    else:
        Qs_lst = [np.zeros_like(dijs) for dijs in dijs_lst]
    Rcuts = get_3c2e_Rcuts(uniq_bas, uniq_basaux, dijs_lst, cell_.vol, omega,
                           precision, fac_type, Qs_lst,
                           eta_correct=eta_correct, R_correct=R_correct,
                           vol_correct=vol_correct_R)
    bas_exps = np.array([np.asarray(b[1:])[:,0].min() for b in uniq_bas])
    atom_Rcuts = get_atom_Rcuts_3c(Rcuts, dijs_lst, bas_exps, uniq_bas_loc,
                                   uniq_basaux_loc)
    cell_rcut = atom_Rcuts.max()
    uniqexp = np.array([np.asarray(b[1:])[:,0].min() for b in uniq_bas])
    nbasauxuniq = len(uniq_basaux)
    dcut2s = dcuts**2.
    Rcut2s = Rcuts**2.
    Ls = get_Lsmin(cell_, atom_Rcuts, uniq_atms)
    prescreening_data = (refuniqshl_map, auxuniqshl_map, nbasauxuniq, uniqexp,
                         dcut2s, dstep_BOHR, Rcut2s, dijs_loc, Ls)
    log.debug("cell rcut for j3c: %.2f", cell_rcut)
    log.debug("keep %d imgs", Ls.shape[0])
    t1 = log.timer_debug1('prescrn warmup', *t1)
# prescreening data ends here

    intor, comp = mol_gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)

    if isinstance(erifile, h5py.Group):
        feri = erifile
    elif h5py.is_hdf5(erifile):
        feri = h5py.File(erifile, 'a')
    else:
        feri = h5py.File(erifile, 'w')
    if dataname in feri:
        del(feri[dataname])
    if dataname+'-kptij' in feri:
        del(feri[dataname+'-kptij'])

    if kptij_lst is None:
        kptij_lst = np.zeros((1,2,3))
    feri[dataname+'-kptij'] = kptij_lst

    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)
    shls_slice_fat = (0, cell_fat.nbas, 0, cell_fat.nbas, 0, auxcell.nbas)

    if shlpr_mask_fat is None:
        n_compact, n_diffuse = cell_fat._nbas_each_set
        shlpr_mask_fat = np.ones((shls_slice_fat[1]-shls_slice_fat[0],
                                  shls_slice_fat[3]-shls_slice_fat[2]),
                                  dtype=np.int8, order="C")
        shlpr_mask_fat[n_compact:,n_compact:] = 0

    ao_loc = cell.ao_loc_nr()
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)[:shls_slice[5]+1]
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    nkptij = len(kptij_lst)

    nii = (ao_loc[shls_slice[1]]*(ao_loc[shls_slice[1]]+1)//2 -
           ao_loc[shls_slice[0]]*(ao_loc[shls_slice[0]]+1)//2)
    nij = ni * nj

    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]
    aosym_ks2 = abs(kpti-kptj).sum(axis=1) < KPT_DIFF_TOL
    j_only = np.all(aosym_ks2)
    #aosym_ks2 &= (aosym[:2] == 's2' and shls_slice[:2] == shls_slice[2:4])
    aosym_ks2 &= aosym[:2] == 's2'

    if j_only and aosym[:2] == 's2':
        assert(shls_slice[2] == 0)
        nao_pair = nii
    else:
        nao_pair = nij

    if gamma_point(kptij_lst):
        dtype = np.double
    else:
        dtype = np.complex128

    buflen = max(8, int(max_memory*.47e6/16/(nkptij*ni*nj*comp)))
    auxdims = aux_loc[shls_slice[4]+1:shls_slice[5]+1] - aux_loc[shls_slice[4]:shls_slice[5]]
    from pyscf.ao2mo.outcore import balance_segs
    auxranges = balance_segs(auxdims, buflen)
    buflen = max([x[2] for x in auxranges])
    buf = np.empty(nkptij*comp*ni*nj*buflen, dtype=dtype)
    buf1 = np.empty_like(buf)
    bufmem = buf.size*16/1024**2.
    if bufmem > max_memory * 0.5:
        raise RuntimeError("Computing 3c2e integrals requires %.2f MB memory, which exceeds the given maximum memory %.2f MB. Try giving PySCF more memory." % (bufmem*2., max_memory))

    int3c = wrap_int3c_spltbas(cell_fat, cell, auxcell, shlpr_mask_fat,
                               prescreening_data,
                               intor, aosym, comp, kptij_lst,
                               bvk_kmesh=bvk_kmesh)

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
# sorted_ij_idx: Sort and group the kptij_lst according to the ordering in
# df._make_j3c to reduce the data fragment in the hdf5 file.  When datasets
# are written to hdf5, they are saved sequentially. If the integral data are
# saved as the order of kptij_lst, removing the datasets in df._make_j3c will
# lead to holes that can not be reused.
    sorted_ij_idx = np.hstack([np.where(uniq_inverse == k)[0]
                                  for k, kpt in enumerate(uniq_kpts)])

    tril_idx = np.tril_indices(ni)
    tril_idx = tril_idx[0] * ni + tril_idx[1]
    def save(istep, mat):
        tspan = np.zeros((2))
        t1 = np.zeros((2))
        t2 = np.zeros((2))
        for k in sorted_ij_idx:
            v = mat[k]
            if gamma_point(kptij_lst[k]):
                v = v.real
            if aosym_ks2[k] and nao_pair == ni**2:
                v = v[:,tril_idx]
            t1[:] = [logger.process_clock(), logger.perf_counter()]
            feri['%s/%d/%d' % (dataname,k,istep)] = v
            t2[:] = [logger.process_clock(), logger.perf_counter()]
            tspan += t2 - t1
        log.debug1("    CPU time for %s %9.2f sec, wall time %9.2f sec",
                   "     save %d"%istep, *tspan)

    with lib.call_in_background(save) as bsave:
        for istep, auxrange in enumerate(auxranges):
            sh0, sh1, nrow = auxrange
            sub_slice = (shls_slice_fat[0], shls_slice_fat[1],
                         shls_slice_fat[2], shls_slice_fat[3],
                         shls_slice_fat[4]+sh0, shls_slice_fat[4]+sh1)
            mat = np.ndarray((nkptij,comp,nao_pair,nrow), dtype=dtype, buffer=buf)
            t1 = (logger.process_clock(), logger.perf_counter())
            bsave(istep, int3c(sub_slice, mat))
            t1 = log.timer_debug1('cmpt+save %d'%istep, *t1)
            buf, buf1 = buf1, buf

    if not isinstance(erifile, h5py.Group):
        feri.close()
    return erifile

def wrap_int3c_spltbas(cell, cell0, auxcell, shlpr_mask, prescreening_data,
                       intor='int3c2e', aosym='s1', comp=1,
                       kptij_lst=np.zeros((1,2,3)),
                       cintopt=None, bvk_kmesh=None
                      ):
    log = logger.Logger(cell.stdout, cell.verbose)

    refuniqshl_map, auxuniqshl_map, nbasauxuniq, uniqexp, dcut2s, dstep_BOHR, Rcut2s, dijs_loc, Ls = prescreening_data

    intor = cell._add_suffix(intor)
    pcell = copy.copy(cell)
    pcell._atm, pcell._bas, pcell._env = \
    atm, bas, env = mol_gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    ao_loc = mol_gto.moleintor.make_loc(bas, intor)
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)
    ao_loc = np.asarray(np.hstack([ao_loc, ao_loc[-1]+aux_loc[1:]]),
                           dtype=np.int32)
    atm, bas, env = mol_gto.conc_env(atm, bas, env,
                                 auxcell._atm, auxcell._bas, auxcell._env)
    nimgs = len(Ls)
    nbas = cell.nbas

    pcell0 = copy.copy(cell0)
    pcell0._atm, pcell0._bas, pcell0._env = \
    atm0, bas0, env0 = mol_gto.conc_env(cell0._atm, cell0._bas, cell0._env,
                                    cell0._atm, cell0._bas, cell0._env)
    ao_loc0 = mol_gto.moleintor.make_loc(bas0, intor)
    ao_loc0 = np.asarray(np.hstack([ao_loc0, ao_loc0[-1]+aux_loc[1:]]),
                            dtype=np.int32)
    atm0, bas0, env0 = mol_gto.conc_env(atm0, bas0, env0,
                                    auxcell._atm, auxcell._bas, auxcell._env)
    nbas0 = cell0.nbas

    # map orig cell to cell_fat
    bas_idx0 = np.asarray(cell._bas_idx)
    shl_idx0 = [np.where(bas_idx0 == ib)[0] for ib in range(cell0.nbas)]
    shl_loc0 = np.cumsum([0] + [s.size for s in shl_idx0])
    shl_idx0 = np.concatenate(shl_idx0)
    shl_loc0 = np.asarray(shl_loc0, dtype=np.int32, order="C")
    shl_idx0 = np.asarray(shl_idx0, dtype=np.int32, order="C")

    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]

    if bvk_kmesh is None:
        Ls_ = Ls
    else:
        raise RuntimeError

    if gamma_point(kptij_lst):
        assert(aosym[:2] == "s2")
        kk_type = 'g'
        dtype = np.double
        nkpts = nkptij = 1
        kptij_idx = np.array([0], dtype=np.int32)
        expkL = np.ones(1)
    elif is_zero(kpti-kptj):  # j_only
        kk_type = 'k'
        dtype = np.complex128
        kpts = kptij_idx = np.asarray(kpti, order='C')
        expkL = np.exp(1j * np.dot(kpts, Ls_.T))
        nkpts = nkptij = len(kpts)
    else:
        kk_type = 'kk'
        dtype = np.complex128
        kpts = unique(np.vstack([kpti,kptj]))[0]
        expkL = np.exp(1j * np.dot(kpts, Ls_.T))
        wherei = np.where(abs(kpti.reshape(-1,1,3)-kpts).sum(axis=2) < KPT_DIFF_TOL)[1]
        wherej = np.where(abs(kptj.reshape(-1,1,3)-kpts).sum(axis=2) < KPT_DIFF_TOL)[1]
        nkpts = len(kpts)
        kptij_idx = np.asarray(wherei*nkpts+wherej, dtype=np.int32)
        nkptij = len(kptij_lst)

    if cintopt is None:
        if nbas > 0:
            cintopt = _vhf.make_cintopt(atm, bas, env, intor)
        else:
            cintopt = lib.c_null_ptr()
# Remove the precomputed pair data because the pair data corresponds to the
# integral of cell #0 while the lattice sum moves shls to all repeated images.
        if intor[:3] != 'ECP':
            libpbc.CINTdel_pairdata_optimizer(cintopt)

    if gamma_point(kptij_lst):
        fill = 'PBCnr3c_gs2_spltbas'
        drv = libpbc.PBCnr3c_g_spltbas_drv

        log.debug("Using %s to evaluate SR integrals", fill)

        def int3c(shls_slice, out):
            shls_slice = (shls_slice[0], shls_slice[1],
                          nbas+shls_slice[2], nbas+shls_slice[3],
                          nbas*2+shls_slice[4], nbas*2+shls_slice[5])
            drv(getattr(libpbc, intor), getattr(libpbc, fill),
                out.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(comp), ctypes.c_int(nimgs),
                Ls.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int*6)(*shls_slice),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int*6)(*shls_slice0),
                ao_loc0.ctypes.data_as(ctypes.c_void_p),
                shl_idx0.ctypes.data_as(ctypes.c_void_p),
                shl_loc0.ctypes.data_as(ctypes.c_void_p),
                cintopt,
                shlpr_mask.ctypes.data_as(ctypes.c_void_p),  # shlpr_mask
                refuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                auxuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbasauxuniq),
                uniqexp.ctypes.data_as(ctypes.c_void_p),
                dcut2s.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_double(dstep_BOHR),
                Rcut2s.ctypes.data_as(ctypes.c_void_p),
                dijs_loc.ctypes.data_as(ctypes.c_void_p),
                atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
                bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbas),  # need to pass cell.nbas to libpbc.PBCnr3c_drv
                env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size),
                bas0.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas0),
                env0.ctypes.data_as(ctypes.c_void_p)
                )
            return out

    else:
        if bvk_kmesh is None:
            fill = 'PBCnr3c_%s%s' % (kk_type, aosym[:2])
            drv = libpbc.PBCnr3c_drv
        else:
            fill = 'PBCnr3c_bvk_%s%s' % (kk_type, aosym[:2])
            drv = libpbc.PBCnr3c_bvk_drv

        log.debug("Using %s to evaluate SR integrals", fill)

    # if bvk_kmesh is None:
    #     raise NotImplementedError
    # else:
    #     def int3c(shls_slice, out):
    #         shls_slice = (shls_slice[0], shls_slice[1],
    #                       nbas+shls_slice[2], nbas+shls_slice[3],
    #                       nbas*2+shls_slice[4], nbas*2+shls_slice[5])
    #         msh_shift = (nbas-nbas0) * 2
    #         shls_slice0 = (0, nbas0, nbas0, nbas0*2,
    #                        shls_slice[4]-msh_shift, shls_slice[5]-msh_shift)
    #         out.fill(0)
    #         # for some (unknown) reason, this line is needed for the c code to use pbcopt->fprescreen
    #         _ = pbcopt._this == lib.c_null_ptr()
    #         drv(getattr(libpbc, intor), getattr(libpbc, fill),
    #             out.ctypes.data_as(ctypes.c_void_p),
    #             ctypes.c_int(nkptij), ctypes.c_int(nkpts),
    #             ctypes.c_int(comp), ctypes.c_int(nimgs),
    #             ctypes.c_int(bvk_nimgs),   # bvk_nimgs
    #             Ls.ctypes.data_as(ctypes.c_void_p),
    #             expkL.ctypes.data_as(ctypes.c_void_p),
    #             kptij_idx.ctypes.data_as(ctypes.c_void_p),
    #             (ctypes.c_int*6)(*shls_slice),
    #             ao_loc.ctypes.data_as(ctypes.c_void_p),
    #             (ctypes.c_int*6)(*shls_slice0),    # shls_slice0
    #             ao_loc0.ctypes.data_as(ctypes.c_void_p),  # ao_loc0
    #             shl_idx0.ctypes.data_as(ctypes.c_void_p),   # shl_idx0
    #             shl_loc0.ctypes.data_as(ctypes.c_void_p),   # shl_loc0
    #             cell_loc_bvk.ctypes.data_as(ctypes.c_void_p),   # cell_loc_bvk
    #             shlpr_mask.ctypes.data_as(ctypes.c_void_p),  # shlpr_mask
    #             cintopt, cpbcopt,
    #             atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
    #             bas.ctypes.data_as(ctypes.c_void_p),
    #             ctypes.c_int(nbas),  # need to pass cell.nbas to libpbc.PBCnr3c_drv
    #             env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size),
    #             bas0.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas0),
    #             env0.ctypes.data_as(ctypes.c_void_p))
    #         return out

    return int3c

""" Helper functions for AFT/FFT ao pairs
"""
# Analytical Fourier transform of AO pairs. Support (1) masking out certain shlprs, and (2) computing using cell_fat (see _reorder_cell above) and write to a buffer according to the original cell.
# NOTE buffer "out" must be initialized to 0
# gxyz is the index for Gvbase
def ft_aopair_kpts_spltbas(cell, cell0, Gv, shls_slice0=None, aosym='s1',
                   b=None, gxyz=None, Gvbase=None, q=np.zeros(3),
                   kptjs=np.zeros((1,3)), intor='GTO_ft_ovlp', comp=1,
                   bvk_kmesh=None, shlpr_mask=None, out=None):
    r'''
    Modified from ft_ao.py, ft_aopair_kpts.
    This function should be solely used by RSDF.

    Fourier transform AO pair for a group of k-points
    \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3

    The return array holds the AO pair
    corresponding to the kpoints given by kptjs
    '''

    intor = cell._add_suffix(intor)

    q = np.reshape(q, 3)
    kptjs = np.asarray(kptjs, order='C').reshape(-1,3)
    Gv = np.asarray(Gv, order='C').reshape(-1,3)
    nGv = Gv.shape[0]
    GvT = np.asarray(Gv.T, order='C')
    GvT += q.reshape(-1,1)

    if (gxyz is None or b is None or Gvbase is None or (abs(q).sum() > 1e-9)
# backward compatibility for pyscf-1.2, in which the argument Gvbase is gs
        or (Gvbase is not None and isinstance(Gvbase[0], (int, np.integer)))):
        p_gxyzT = lib.c_null_ptr()
        p_mesh = (ctypes.c_int*3)(0,0,0)
        p_b = (ctypes.c_double*1)(0)
        eval_gz = 'GTO_Gv_general'
    else:
        if abs(b-np.diag(b.diagonal())).sum() < 1e-8:
            eval_gz = 'GTO_Gv_orth'
        else:
            eval_gz = 'GTO_Gv_nonorth'
        gxyzT = np.asarray(gxyz.T, order='C', dtype=np.int32)
        p_gxyzT = gxyzT.ctypes.data_as(ctypes.c_void_p)
        b = np.hstack((b.ravel(), q) + Gvbase)
        p_b = b.ctypes.data_as(ctypes.c_void_p)
        p_mesh = (ctypes.c_int*3)(*[len(x) for x in Gvbase])

    Ls = cell.get_lattice_Ls()
    Ls = Ls[np.linalg.norm(Ls, axis=1).argsort()]
    nkpts = len(kptjs)
    nimgs = len(Ls)
    nbas = cell.nbas
    nbas0 = cell0.nbas

    # map orig cell to cell_fat
    bas_idx0 = np.asarray(cell._bas_idx)
    shl_idx0 = [np.where(bas_idx0 == ib)[0] for ib in range(cell0.nbas)]
    shl_loc0 = np.cumsum([0] + [s.size for s in shl_idx0])
    shl_idx0 = np.concatenate(shl_idx0)
    shl_loc0 = np.asarray(shl_loc0, dtype=np.int32, order="C")
    shl_idx0 = np.asarray(shl_idx0, dtype=np.int32, order="C")

    # determine shlpr mask
    if shlpr_mask is None:
        n_compact, n_diffuse = cell._nbas_each_set
        shlpr_mask = np.ones((nbas, nbas), dtype=np.int8, order="C")
        shlpr_mask[n_compact:,n_compact:] = 0

    if bvk_kmesh is None:
        raise NotImplementedError
        expkL = np.exp(1j * np.dot(kptjs, Ls.T))
    else:
        from pyscf.pbc.df.ft_ao import _estimate_overlap
        ovlp_mask = _estimate_overlap(cell, Ls) > cell.precision
        ovlp_mask = np.asarray(ovlp_mask, dtype=np.int8, order='C')

        # Using Ls = translations.dot(a)
        translations = np.linalg.solve(cell.lattice_vectors().T, Ls.T)
        # t_mod is the translations inside the BvK cell
        t_mod = translations.round(3).astype(int) % np.asarray(bvk_kmesh)[:,None]
        cell_loc_bvk = np.ravel_multi_index(t_mod, bvk_kmesh).astype(np.int32)

        bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, bvk_kmesh)
        expkL = np.exp(1j * np.dot(kptjs, bvkmesh_Ls.T))

    atm, bas, env = mol_gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    ao_loc = mol_gto.moleintor.make_loc(bas, intor)
    atm0, bas0, env0 = mol_gto.conc_env(cell0._atm, cell0._bas, cell0._env,
                                    cell0._atm, cell0._bas, cell0._env)
    ao_loc0 = mol_gto.moleintor.make_loc(bas0, intor)
    if shls_slice0 is None:
        shls_slice0 = (0, nbas0, nbas0, nbas0*2)
    else:
        shls_slice0 = (shls_slice0[0], shls_slice0[1],
                      nbas0+shls_slice0[2], nbas0+shls_slice0[3])
    shls_slice = (0, nbas, nbas, nbas*2)
    ni = ao_loc0[shls_slice0[1]] - ao_loc0[shls_slice0[0]]
    nj = ao_loc0[shls_slice0[3]] - ao_loc0[shls_slice0[2]]
    shape = (nkpts, comp, ni, nj, nGv)

# Theoretically, hermitian symmetry can be also found for kpti == kptj:
#       f_ji(G) = \int f_ji exp(-iGr) = \int f_ij^* exp(-iGr) = [f_ij(-G)]^*
# hermi operation needs reordering the axis-0.  It is inefficient.
    if aosym == 's1hermi': # Symmetry for Gamma point
        assert(is_zero(q) and is_zero(kptjs) and ni == nj)
    elif aosym == 's2':
        i0 = ao_loc0[shls_slice0[0]]
        i1 = ao_loc0[shls_slice0[1]]
        nij = i1*(i1+1)//2 - i0*(i0+1)//2
        shape = (nkpts, comp, nij, nGv)

    cintor = getattr(libpbc, intor)
    eval_gz = getattr(libpbc, eval_gz)

    out = np.ndarray(shape, dtype=np.complex128, buffer=out)
    out.fill(0)

    if bvk_kmesh is None:
        raise NotImplementedError
        if nkpts == 1:
            fill = getattr(libpbc, 'PBC_ft_fill_nk1'+aosym)
        else:
            fill = getattr(libpbc, 'PBC_ft_fill_k'+aosym)
        drv = libpbc.PBC_ft_latsum_drv
        drv(cintor, eval_gz, fill, out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nkpts), ctypes.c_int(comp), ctypes.c_int(nimgs),
            Ls.ctypes.data_as(ctypes.c_void_p), expkL.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*4)(*shls_slice), ao_loc.ctypes.data_as(ctypes.c_void_p),
            GvT.ctypes.data_as(ctypes.c_void_p), p_b, p_gxyzT, p_mesh, ctypes.c_int(nGv),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.nbas),
            env.ctypes.data_as(ctypes.c_void_p))
    else:
        if nkpts == 1:
            fill_name = 'PBC_ft_bvk_nk1'+aosym+"_spltbas"
        else:
            fill_name = 'PBC_ft_bvk_k'+aosym+"_spltbas"
        fill = getattr(libpbc, fill_name)
        drv = libpbc.PBC_ft_bvk_spltbas_drv
        drv(cintor, eval_gz, fill, out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nkpts), ctypes.c_int(comp),
            ctypes.c_int(nimgs), ctypes.c_int(expkL.shape[1]),
            Ls.ctypes.data_as(ctypes.c_void_p), expkL.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*4)(*shls_slice), ao_loc.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*4)(*shls_slice0), ao_loc0.ctypes.data_as(ctypes.c_void_p),
            shl_idx0.ctypes.data_as(ctypes.c_void_p),   # shl_idx0
            shl_loc0.ctypes.data_as(ctypes.c_void_p),   # shl_loc0
            shlpr_mask.ctypes.data_as(ctypes.c_void_p),  # shlpr_mask
            cell_loc_bvk.ctypes.data_as(ctypes.c_void_p),
            ovlp_mask.ctypes.data_as(ctypes.c_void_p),
            GvT.ctypes.data_as(ctypes.c_void_p), p_b, p_gxyzT, p_mesh, ctypes.c_int(nGv),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.nbas),
            env.ctypes.data_as(ctypes.c_void_p),
            bas0.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas0),
            env0.ctypes.data_as(ctypes.c_void_p))

    if aosym == 's1hermi':
        for i in range(1,ni):
            out[:,:,:i,i] = out[:,:,i,:i]
    out = np.rollaxis(out, -1, 2)
    if comp == 1:
        out = out[:,0]
    return out


def fft_aopair_kpts_spltbas(nint, cell, cell0, mesh, coords, aosym="s1",
                            q=np.zeros(3), kptjs=np.zeros((1,3)),
                            shls_slice0=None, shl_mask=None, out=None):
    '''Modified from fft_ao2mo.py, get_ao_pairs_G
    This function should be solely used by RSDF.

    Calculate forward (G|ij) FFT of all AO pairs.

    Returns:
        ao_pairs_G : 2D complex array
            For gamma point, the shape is (ngrids, nao*(nao+1)/2); otherwise the
            shape is (ngrids, nao*nao)
    '''
    dtype = np.complex128
    dlen = 16

    q = np.reshape(q, 3)
    if aosym == "s2":
        assert(is_zero(q))
    kptjs = np.reshape(kptjs, (-1,3))
    nkpts = len(kptjs)
    ngrids = coords.shape[0]
    weight = cell.vol / ngrids

    ao_loc = cell.ao_loc_nr()
    nao = ao_loc[-1]
    nbas = ao_loc.size - 1
    ao_loc0 = cell0.ao_loc_nr()
    nao0 = ao_loc0[-1]
    nbas0 = ao_loc0.size - 1

    nbas_c, nbas_d = cell._nbas_each_set
    nao_c = ao_loc[nbas_c]
    nao_d = nao - nao_c
    ao_loc_d = ao_loc[nbas_c:] - nao_c

    if shls_slice0 is None:
        shls_slice0 = (0,nbas0,0,nbas0)

    nish0 = shls_slice0[1] - shls_slice0[0]
    njsh0 = shls_slice0[3] - shls_slice0[2]

    ish0, ish1, jsh0, jsh1 = shls_slice0
    nish0 = ish1 - ish0
    njsh0 = jsh1 - jsh0

    if shl_mask is None:
        shl_mask = np.ones(nbas, dtype=np.bool)
        shl_mask[:nbas_c] = False

    shl_idx = cell._bas_idx
    shls_slice = (nbas_c, nbas)
    bas_idx0 = np.asarray(cell._bas_idx)[shl_mask]  # active shl only
    # map cell0 shl idx to cell0 ao idx
    shl0_to_ao0 = [np.arange(*ao_loc0[ib0:ib0+2]) for ib0 in range(nbas0)]
    # map cell0 shl idx to cell active shl idx
    shl0_to_shl = [np.where(bas_idx0 == ib0)[0] for ib0 in range(nbas0)]

    kjaos = nint.eval_ao(cell, coords, kptjs, shls_slice=shls_slice)
    kjaos = [np.asarray(x.T, order="C") for x in kjaos]

    if aosym == "s2":   # q = 0
        i0, i1 = ao_loc0[ish0], ao_loc0[ish1]
        nao_pair0 = i1*(i1+1)//2 - i0*(i0+1)//2
        ao_pairs_G = np.ndarray((nkpts,nao_pair0,ngrids), dtype=dtype,
                                buffer=out)
        ao_pairs_G.fill(0)

        i0 = ao_loc0[ish0]
        iap0_shift = i0*(i0+1) // 2
        nao02 = nao0*(nao0+1) // 2
        iap0_tab = np.zeros((nao0,nao0), dtype=int)
        iap0_tab[np.tril_indices_from(iap0_tab)] = np.arange(nao02)
        iap0_tab -= iap0_shift

        for k in range(nkpts):
            aoi = kjaos[k].conj() * weight
            aoj = kjaos[k]
            for ib0 in range(ish0,ish1):
                ibs = shl0_to_shl[ib0]
                if ibs.size == 0:
                    continue
                i00,i01 = ao_loc0[ib0], ao_loc0[ib0+1]
                for jb0 in range(ib0+1):
                    jbs = shl0_to_shl[jb0]
                    if jbs.size == 0:
                        continue
                    j00,j01 = ao_loc0[jb0], ao_loc0[jb0+1]
                    idx0 = iap0_tab[i00:i01,j00:j01]
                    tmp = 0
                    for ib in ibs:
                        i0,i1 = ao_loc_d[ib], ao_loc_d[ib+1]
                        for jb in jbs:
                            j0,j1 = ao_loc_d[jb], ao_loc_d[jb+1]
                            tmp += aoi[i0:i1][:,None,:] * aoj[j0:j1]
                    if ib0 == jb0:
                        nj0 = idx0.shape[1]
                        tril_idx = np.tril_indices(nj0)
                        idx0 = idx0[tril_idx].ravel()
                        tmp = tmp[tril_idx]
                    else:
                        idx0 = idx0.ravel()
                        tmp = np.reshape(tmp, (-1,ngrids))
                    ao_pairs_G[k,idx0,] = pbctools.fft(tmp, mesh)
                    tmp = None
    else:
        kptis = kptjs - q
        kiaos = nint.eval_ao(cell, coords, kptis, shls_slice=shls_slice)
        kiaos = [np.asarray(x.T, order="C") for x in kiaos]
        phase = np.exp(-1j * np.dot(coords, q))

        i0, i1 = ao_loc0[ish0], ao_loc0[ish1]
        j0, j1 = ao_loc0[jsh0], ao_loc0[jsh1]
        nao_pair0 = (i1-i0) * (j1-j0)
        ao_pairs_G = np.ndarray((nkpts,nao_pair0,ngrids), dtype=dtype,
                                buffer=out)
        ao_pairs_G.fill(0)

        iap0_tab = np.arange(nao0*nao0, dtype=int).reshape(nao0,nao0)
        iap0_tab -= i0 * nao0

        for k in range(nkpts):
            aoi = kiaos[k].conj() * phase * weight
            aoj = kjaos[k]
            for ib0 in range(ish0,ish1):
                ibs = shl0_to_shl[ib0]
                if ibs.size == 0:
                    continue
                i00,i01 = ao_loc0[ib0], ao_loc0[ib0+1]
                for jb0 in range(jsh0,jsh1):
                    jbs = shl0_to_shl[jb0]
                    if jbs.size == 0:
                        continue
                    j00,j01 = ao_loc0[jb0], ao_loc0[jb0+1]
                    idx0 = iap0_tab[i00:i01,j00:j01].ravel()
                    tmp = 0.
                    for ib in ibs:
                        i0,i1 = ao_loc_d[ib], ao_loc_d[ib+1]
                        for jb in jbs:
                            j0,j1 = ao_loc_d[jb], ao_loc_d[jb+1]
                            tmp += aoi[i0:i1][:,None,:] * aoj[j0:j1]
                    tmp = np.reshape(tmp, (-1,ngrids))
                    ao_pairs_G[k,idx0] = pbctools.fft(tmp, mesh)
                    tmp = None

    ao_pairs_G = np.rollaxis(ao_pairs_G, -1, 1)

    return ao_pairs_G


###
def fat_orig_loop(cell_fat, cell, aosym='s2', shlpr_mask=None, verbose=0):
    if aosym == 's2':
        return _fat_orig_loop_s2(cell_fat, cell, shlpr_mask, verbose)
    elif aosym == 's1':
        return _fat_orig_loop_s1(cell_fat, cell, shlpr_mask, verbose)
    else:
        raise KeyError('Unknown aosym = %s' % aosym)


def _unfold_cgto_shl(cell_):
    ao_loc_old = cell_.ao_loc_nr()
    ao_loc = []
    bas_map = []
    for ib in range(cell_.nbas):
        nctr = cell_.bas_nctr(ib)
        i0, i1 = ao_loc_old[ib:ib+2]
        norb = i1 - i0
        ndeg = norb // nctr
        assert(norb % nctr == 0)
        ao_loc.append(np.arange(i0, i0+norb, ndeg))
        bas_map += [ib,] * nctr
    ao_loc.append([ao_loc_old[-1]])
    ao_loc = np.concatenate(ao_loc)

    return ao_loc, np.asarray(bas_map)


def _fat_orig_loop_s2(cell_fat, cell, shlpr_mask=False, verbose=0):

    log = logger.Logger(cell.stdout, verbose)

    # For aosym='s2', we need to unravel cGTOs that share same exponents
    ao_loc, bas_map = _unfold_cgto_shl(cell_fat)
    nbas = ao_loc.size - 1
    nao = ao_loc[-1]

    ao_loc0, bas_map0 = _unfold_cgto_shl(cell)
    nbas0 = ao_loc0.size - 1
    nao0 = ao_loc0[-1]

    ao_loc_fat = cell_fat.ao_loc_nr()
    nbas_c_fat, nbas_d_fat = cell_fat._nbas_each_set
    nbas_c = np.where(ao_loc <= ao_loc_fat[nbas_c_fat])[0][-1]
    nbas_d = nbas - nbas_c

    bas_idx_old = cell_fat._bas_idx
    bas_idx = []
    for ib_old in range(cell_fat.nbas):
        ib_lst = np.where(bas_map == ib_old)[0]
        ib0_lst = np.where(bas_map0 == bas_idx_old[ib_old])[0]
        bas_idx.append(ib0_lst)
    bas_idx = np.concatenate(bas_idx)

    bas_f2o = bas_idx
    log.debug2("bas_f2o: %s", bas_f2o)
    log.debug2("ao_loc: %s", ao_loc)
    log.debug2("ao_loc0: %s", ao_loc0)

    ao_rg = [np.arange(*ao_loc[i:i+2]) for i in range(nbas)]
    ao_rg0 = [np.arange(*ao_loc0[i:i+2]) for i in range(nbas0)]
    log.debug2("ao range list= %s", ao_rg)
    log.debug2("ao range list0= %s", ao_rg0)

    if shlpr_mask is None:
        shlpr_mask = np.ones((nbas,nbas), dtype=int)

    A = np.zeros((nao,nao), dtype=int)
    nao2 = nao*(nao+1) // 2
    A[np.tril_indices(nao)] = np.arange(nao2)
    A = A.ravel()

    A0_ieqj = np.zeros((nao0,nao0), dtype=int)
    nao02 = nao0*(nao0+1) // 2
    A0_ieqj[np.tril_indices(nao0)] = np.arange(nao02)
    A0_ineqj = np.zeros_like(A0_ieqj)
    A0_ineqj[np.tril_indices(nao0)] = A0_ieqj[np.tril_indices(nao0)]
    A0_ineqj += A0_ineqj.T
    np.fill_diagonal(A0_ineqj, np.diag(A0_ieqj))
    A0_ieqj = A0_ieqj.ravel()
    A0_ineqj = A0_ineqj.ravel()

    for i in range(nbas):
        i0 = bas_f2o[i]
        for j in range(i+1):
            if not shlpr_mask[i,j]:
                continue

            j0 = bas_f2o[j]
            ap = ao_rg[i][:,None]*nao + ao_rg[j]
            ap0 = ao_rg0[i0][:,None]*nao0 + ao_rg0[j0]
            log.debug2("(i,j,i0,j0)= (%d,%d,%d,%d) ap= %s ap0= %s",
                       i,j,i0,j0, ap, ap0)

            # source indices
            if i == j:
                ap = ap[np.tril_indices_from(ap)]
                ap = A[ap]
                ap2 = None
            elif i0 != j0:
                ap = ap.ravel()
                ap = A[ap]
                ap2 = None
            else:   # special case: different sources but same destination
                tril_inds = np.tril_indices_from(ap)
                ap2 = ap.T[tril_inds]
                ap = ap[tril_inds]
                ap = A[ap]
                ap2 = A[ap2]

            # destination indices
            if i0 == j0:
                ap0 = ap0[np.tril_indices_from(ap0)]
                ap0 = A0_ieqj[ap0]
            else:
                ap0 = ap0.ravel()
                ap0 = A0_ineqj[ap0]

            log.debug2("                           ap= %s ap0= %s", ap, ap0)

            yield ap, ap0, ap2


def _fat_orig_loop_s1(cell_fat, cell, shlpr_mask=None, verbose=0):
    log = logger.Logger(sys.stdout, verbose)

    nbas = cell_fat.nbas
    nbas0 = cell.nbas
    ao_loc = cell_fat.ao_loc_nr()
    nao = ao_loc[-1]
    ao_loc0 = cell.ao_loc_nr()
    nao0 = ao_loc0[-1]
    nbas_c, nbas_d = cell_fat._nbas_each_set

    bas_f2o = cell_fat._bas_idx
    log.debug2("bas_f2o: %s", bas_f2o)
    log.debug2("ao_loc: %s", ao_loc)
    log.debug2("ao_loc0: %s", ao_loc0)

    ao_rg = [np.arange(*ao_loc[i:i+2]) for i in range(nbas)]
    ao_rg0 = [np.arange(*ao_loc0[i:i+2]) for i in range(nbas0)]
    log.debug2("ao range list= %s", ao_rg)
    log.debug2("ao range list0= %s", ao_rg0)

    if shlpr_mask is None:
        shlpr_mask = np.ones((nbas,nbas), dtype=int)

    for i in range(nbas):
        i0 = bas_f2o[i]
        for j in range(nbas):
            if not shlpr_mask[i,j]:
                continue

            j0 = bas_f2o[j]
            ap = (ao_rg[i][:,None]*nao + ao_rg[j]).ravel()
            ap0 = (ao_rg0[i0][:,None]*nao0 + ao_rg0[j0]).ravel()

            yield ap, ap0
