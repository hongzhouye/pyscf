
#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#         Gengzhi Yang <genzyang17@gmail.com>
#


import ctypes
import numpy as np

from pyscf import gto
from pyscf import lib
from pyscf.scf import _vhf
from pyscf.pbc.lo.base import get_kmesh
from pyscf.pbc.tdm import truncation


libpbc = lib.load_library('libpbc')


def get_k(mytdm, dm, hermi=1, kpts=None, kpts_band=None, omega=None):
    if kpts_band is not None:
        raise NotImplementedError('TDM does not support band k-points')
    if omega not in (None, 0):
        raise NotImplementedError('TDM does not support range-separated K')
    if kpts is None:
        kpts = mytdm.kpts

    cell = mytdm.cell
    kpts = np.reshape(kpts, (-1, 3))
    kmesh = get_kmesh(cell, kpts)
    nkpts = len(kpts)
    nao = cell.nao_nr()

    dm = np.asarray(dm)
    dm_shape = dm.shape
    if dm_shape[-3:] != (nkpts, nao, nao):
        raise ValueError('Invalid density matrix shape %s' % (dm_shape,))
    dms = dm.reshape(-1, nkpts, nao, nao)

    dm_Ls, atmweights = truncation.get_dm_Ls(
        cell, kpts, level=mytdm.dm_trunc_level,
        shape=mytdm.dm_trunc_shape, ws_weight=mytdm.ws_weight,
        rcut=mytdm.dm_rcut, search_mesh=mytdm.ws_search_mesh)
    eri_Ls = _get_schwarz_Ls(cell)[0]

    log = lib.logger.new_logger(mytdm)
    log.info('')
    log.info('TDM K build')
    log.info('DM truncation = %s-%s',
             mytdm.dm_trunc_level, mytdm.dm_trunc_shape)
    if mytdm.dm_trunc_shape == 'ws':
        log.info('WS boundary weights = %s', mytdm.ws_weight)
    log.debug('DM lattice cells = %d', len(dm_Ls))
    log.debug('ERI lattice cells = %d', len(eri_Ls))
    log.debug1('DM lattice vectors =\n%s', dm_Ls)

    vks = []
    for dm_kpts in dms:
        phase = np.exp(-1j*np.dot(kpts, dm_Ls.T))
        dm_real = _k_to_real(dm_kpts, phase)
        dm_real = truncation.apply_weights(cell, atmweights, dm_real)
        vks.append(_contract_k(
            cell, kpts, kmesh, eri_Ls, dm_Ls, dm_real,
            mytdm.direct_scf_tol, mytdm.extent_tol,
            mytdm.profile, mytdm.verbose))

    return np.asarray(vks).reshape(dm_shape)


def _k_to_real(a_kpts, phase, imag_tol=1e-4):
    nkpts = len(a_kpts)
    a_real = np.einsum('kpq,kR->Rpq', a_kpts, phase) / nkpts
    real_norm = max(np.linalg.norm(a_real.real), 1e-30)
    if abs(a_real.imag).max() > imag_tol * real_norm:
        raise RuntimeError('TDM requires a real density matrix in real space')
    return np.asarray(a_real.real, order='C')


def _contract_k(cell, kpts, kmesh, eri_Ls, dm_Ls, dm_real,
                direct_scf_tol, extent_tol, profile=False, verbose=None):
    log = lib.logger.new_logger(cell, verbose)
    cpu0 = (lib.logger.process_clock(), lib.logger.perf_counter())

    ao_loc0 = cell.ao_loc_nr()
    nao = cell.nao_nr()
    lattice_vectors = cell.lattice_vectors()

    atm, bas, env = gto.conc_env(
        cell._atm, cell._bas, cell._env,
        cell._atm, cell._bas, cell._env)
    atm, bas, env = gto.conc_env(atm, bas, env, atm, bas, env)

    intor = gto.moleintor._get_intor_and_comp(
        cell._add_suffix('int2e'), None)[0]
    fintor = getattr(gto.moleintor.libcgto, intor)
    cintopt = _vhf.make_cintopt(atm, bas, env, intor)
    libpbc.CINTdel_pairdata_optimizer(cintopt)
    ao_loc = gto.moleintor.make_loc(bas, intor)

    as_double = {'dtype': np.float64, 'order': 'C'}
    as_int = {'dtype': np.int32, 'order': 'C'}

    bvk_ncells = int(np.prod(kmesh))
    bvk_Ts = lib.cartesian_prod([np.arange(x) for x in kmesh])
    bvk_Ls = np.dot(bvk_Ts, lattice_vectors)

    eri_Ls, bvk_cell_loc = _sort_Ls_bvk(cell, eri_Ls, kmesh)
    eri_Ls = np.asarray(eri_Ls, **as_double)
    bvk_cell_loc = np.asarray(bvk_cell_loc, **as_int)

    t_mod = (bvk_Ts[:,None] + bvk_Ts).reshape(-1, 3).T
    t_mod %= np.asarray(kmesh)[:,None]
    bvkadd_loc = np.ravel_multi_index(t_mod, kmesh)
    bvkadd_loc = np.asarray(bvkadd_loc, **as_int)

    trans = np.linalg.solve(
        lattice_vectors.T, (dm_Ls[:,None] + eri_Ls).reshape(-1, 3).T)
    t_mod = trans.round(3).astype(int)
    t_mod %= np.asarray(kmesh)[:,None]
    bvkidx_by_dmcell = np.ravel_multi_index(t_mod, kmesh)
    bvkidx_by_dmcell = bvkidx_by_dmcell.reshape(len(dm_Ls), len(eri_Ls))
    bvkidx_by_dmcell = np.asarray(bvkidx_by_dmcell, **as_int)

    log.timer('TDM BvK indices', *cpu0)

    dm_real = np.asarray(dm_real, **as_double)
    dm_cond = np.asarray([
        lib.condense('NP_absmax', x, ao_loc0) for x in dm_real])
    dm_cond = np.asarray(dm_cond.transpose(1, 2, 0), **as_double)
    dm_Ls = np.asarray(dm_Ls, **as_double)

    if profile:
        log.info('TDM profile: nkpts = %d, nbas = %d, nao = %d',
                 bvk_ncells, cell.nbas, nao)
        log.info('TDM profile: DM cells = %d, ERI cells = %d, threads = %d',
                 len(dm_Ls), len(eri_Ls), lib.num_threads())
        log.info('TDM profile: direct_scf_tol = %.1e', direct_scf_tol)

    wall0 = lib.logger.perf_counter()
    q_cond = _precompute_q_cond(cell, eri_Ls)
    if profile:
        log.info('TDM profile: q_cond wall time = %.3f sec',
                 lib.logger.perf_counter() - wall0)
    wall0 = lib.logger.perf_counter()
    ext_cond, r_cond = _precompute_extent(cell, eri_Ls, extent_tol)
    if profile:
        log.info('TDM profile: extent wall time = %.3f sec',
                 lib.logger.perf_counter() - wall0)
    log.timer('TDM integral screening', *cpu0)

    vk_bvk = np.zeros((bvk_ncells, nao, nao), **as_double)
    args = (
        fintor,
        vk_bvk.ctypes.data_as(ctypes.c_void_p), cintopt,
        ctypes.c_int(len(dm_Ls)),
        dm_Ls.ctypes.data_as(ctypes.c_void_p),
        dm_real.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(len(eri_Ls)),
        eri_Ls.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(len(bvk_Ls)),
        bvk_cell_loc.ctypes.data_as(ctypes.c_void_p),
        bvkadd_loc.ctypes.data_as(ctypes.c_void_p),
        bvkidx_by_dmcell.ctypes.data_as(ctypes.c_void_p),
        q_cond.ctypes.data_as(ctypes.c_void_p),
        ext_cond.ctypes.data_as(ctypes.c_void_p),
        r_cond.ctypes.data_as(ctypes.c_void_p),
        dm_cond.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(direct_scf_tol),
        ao_loc.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm*4),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.nbas*4),
        env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size))

    wall0 = lib.logger.perf_counter()
    if profile:
        counts = np.zeros(11, dtype=np.uint64)
        times = np.zeros(3)
        libpbc.PBCtdm_k_drv_profile(
            *args, counts.ctypes.data_as(ctypes.c_void_p),
            times.ctypes.data_as(ctypes.c_void_p))
        _log_profile(log, counts, times,
                     lib.logger.perf_counter() - wall0)
    else:
        libpbc.PBCtdm_k_drv(*args)

    phase = np.exp(1j*np.dot(kpts, bvk_Ls.T))
    vk = np.einsum('kR,Rpq->kpq', phase, vk_bvk)
    log.timer('TDM K build', *cpu0)
    return vk


def _log_profile(log, counts, times, wall_time):
    def screening(label, tested, skipped):
        kept = tested - skipped
        fraction = kept/tested if tested else 0
        log.info('TDM profile: %-4s kept %d / %d (%.1f%%)',
                 label, kept, tested, fraction*100)

    log.info('TDM profile: C driver wall time = %.3f sec', wall_time)
    screening('DM', counts[0], counts[1])
    screening('bra', counts[2], counts[3])
    screening('ket', counts[4], counts[5])
    screening('QQR', counts[6], counts[7])
    screening('ERI', counts[8], counts[8]-counts[9])
    log.info('TDM profile: contraction calls = %d', counts[10])
    log.info('TDM profile: thread time = %.3f sec', times[0])
    log.info('TDM profile: integral time = %.3f sec', times[1])
    log.info('TDM profile: contraction time = %.3f sec', times[2])
    log.info('TDM profile: other/idle time = %.3f sec',
             max(times[0]-times[1]-times[2], 0))


def _precompute_q_cond(cell, Ls):
    atm, bas, env = gto.conc_env(
        cell._atm, cell._bas, cell._env,
        cell._atm, cell._bas, cell._env)

    drv = libpbc.PBCtdm_q_cond
    intor = gto.moleintor._get_intor_and_comp(
        cell._add_suffix('int2e'), None)[0]
    fintor = getattr(gto.moleintor.libcgto, intor)
    cintopt = lib.c_null_ptr()

    q_cond = np.empty((cell.nbas, cell.nbas, len(Ls)), order='C')
    ao_loc = gto.moleintor.make_loc(bas, intor)
    drv(fintor, q_cond.ctypes.data_as(ctypes.c_void_p), cintopt,
        ctypes.c_int(len(Ls)), Ls.ctypes.data_as(ctypes.c_void_p),
        ao_loc.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm*2),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.nbas*2),
        env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size))
    return q_cond


def _precompute_extent(cell, Ls, threshold):
    from scipy.special import erfcinv

    mol = cell.to_mol()
    bas_exp = [mol.bas_exp(i) for i in range(mol.nbas)]
    bas_coeff = [abs(mol._libcint_ctr_coeff(i)).max(axis=1)
                 for i in range(mol.nbas)]
    bas_coords = [mol.bas_coord(i) for i in range(mol.nbas)]
    erfcinvfac = erfcinv(threshold)

    extent = np.zeros((mol.nbas, mol.nbas, len(Ls)), order='C')
    center = np.zeros((mol.nbas, mol.nbas, len(Ls), 3), order='C')
    for ib,(ei,ci,ri) in enumerate(zip(bas_exp, bas_coeff, bas_coords)):
        for jb,(ej,cj,rj) in enumerate(
                zip(bas_exp, bas_coeff, bas_coords)):
            eij = ei[:,None] + ej
            cij = ci[:,None] * cj
            extent_ij = np.sqrt(2./eij) * erfcinvfac
            rij = ((ei[:,None]*ri)[:,None,None,:]
                   + ej[:,None,None]*(rj+Ls)) / eij[:,:,None,None]
            center[ib,jb] = (
                cij[:,:,None,None]*rij).sum(axis=(0, 1)) / cij.sum()
            extent[ib,jb] = np.max(
                extent_ij[:,:,None]
                + lib.norm(rij-center[ib,jb], axis=-1), axis=(0, 1))
    return extent, center


def _get_ovlp_dcut(basis, precision, r0=30):
    from pyscf.pbc.df.rsdf_helper import _binary_search

    mol = gto.M(atom='H 0 0 0; H 0 0 0', basis=basis)
    nbas = mol.nbas // 2
    es = np.asarray([mol.bas_exp(i).min() for i in range(nbas)])
    etas = 1/(1/es[:,None] + 1/es)

    dcuts = np.zeros((nbas, nbas))
    for i in range(nbas):
        for j in range(i+1):
            shls_slice = (i, i+1, nbas+j, nbas+j+1)
            precision_ij = precision * min(etas[i,j], 1.)

            def overlap_below_threshold(r):
                mol._env[mol._atm[1,gto.PTR_COORD]] = r
                overlap = np.linalg.norm(
                    mol.intor('int1e_ovlp', shls_slice=shls_slice))
                return overlap < precision_ij * min(1./r, 1.)

            dcuts[i,j] = dcuts[j,i] = _binary_search(
                r0*.3, r0, 1, True, overlap_below_threshold)
    return dcuts


def _get_rcut_atoms(cell, precision=None):
    if precision is None:
        precision = cell.precision

    atoms = np.asarray([cell.atom_symbol(i) for i in range(cell.natm)])
    unique_atoms, unique_idx, unique_inv = np.unique(
        atoms, return_index=True, return_inverse=True)
    aoslices = cell.aoslice_by_atom()[:,:2]

    unique_slices = []
    offset = 0
    for ia in unique_idx:
        shell_slice = aoslices[ia]
        unique_slice = shell_slice - shell_slice[0] + offset
        offset = unique_slice[1]
        unique_slices.append(unique_slice)

    basis = [b for atom in unique_atoms for b in cell._basis[atom]]
    shell_rcut = _get_ovlp_dcut(basis, precision)
    atom_rcut = np.zeros((len(unique_atoms), len(unique_atoms)))
    for ia,(i0,i1) in enumerate(unique_slices):
        for ja in range(ia+1):
            j0, j1 = unique_slices[ja]
            atom_rcut[ia,ja] = atom_rcut[ja,ia] = \
                shell_rcut[i0:i1,j0:j1].max()

    return atom_rcut[unique_inv[:,None], unique_inv]


def _get_schwarz_Ls(cell, precision=None):
    atom_rcut = _get_rcut_atoms(cell, precision)
    nimgs = truncation.get_nimgs(cell, atom_rcut.max())
    Ts = lib.cartesian_prod([np.arange(-x, x+1) for x in nimgs])
    Ls = np.dot(Ts, cell.lattice_vectors())
    atom_mask = truncation._get_atmdist(cell, Ls) < atom_rcut
    idx = np.any(atom_mask, axis=(1, 2))
    return Ls[idx], atom_mask[idx]


def _sort_Ls_bvk(cell, Ls, kmesh):
    translations = np.linalg.solve(cell.lattice_vectors().T, Ls.T)
    t_mod = translations.round(3).astype(int)
    t_mod %= np.asarray(kmesh)[:,None]
    bvk_idx = np.ravel_multi_index(t_mod, kmesh)

    order = np.argsort(bvk_idx, kind='stable')
    counts = np.bincount(bvk_idx, minlength=np.prod(kmesh))
    cell_loc = np.append(0, np.cumsum(counts)).astype(np.int32)
    return np.asarray(Ls[order], order='C'), cell_loc
