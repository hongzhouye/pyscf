#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#         James McClain <jdmcclain47@gmail.com>
#         Xing Zhang <zhangxing.nju@gmail.com>
#         Hong-Zhou Ye <hzyechem@gmail.com>
#


'''
kpoint-adapted and spin-adapted MP2
t2[i,j,a,b] = <ij|ab> / D_ij^ab

t2 and eris are never stored in full, only a partial
eri of size (nkpts,nocc,nocc,nvir,nvir)
'''

import numpy as np
from scipy.linalg import block_diag
import h5py
import tempfile
import ctypes

from pyscf import lib
from pyscf.lib import logger, einsum
from pyscf.mp import mp2
from pyscf.pbc.df import df
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.lib import kpts as libkpts
from pyscf.lib.parameters import LARGE_DENOM
from pyscf import __config__

libmp = lib.load_library('libmp')

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)


def kernel(mp, mo_energy, mo_coeff, eris=None, with_t2=WITH_T2, verbose=None):
    """Computes k-point RMP2 energy.

    Args:
        mp (KMP2): an instance of KMP2
        mo_energy (list): a list of np.ndarray. Each array contains MO energies of
                          shape (Nmo,) for one kpt
        mo_coeff (list): a list of np.ndarray. Each array contains MO coefficients
                         of shape (Nao, Nmo) for one kpt
        verbose (int, optional): level of verbosity. Defaults to logger.NOTE (=3).
        with_t2 (bool, optional): whether to compute t2 amplitudes. Defaults to WITH_T2 (=True).

    Returns:
        KMP2 energy and t2 amplitudes (=None if with_t2 is False)
    """
    log = logger.new_logger(mp, verbose=verbose)
    if mp._kernel is None:
        if mp.with_df_ints:
            fkernel = kernel_df if with_t2 else kernel_df_C
        else:
            fkernel = kernel_fftdf
    elif callable(mp._kernel):
        fkernel = mp._kernel
    elif isinstance(mp._kernel, str):
        if mp.with_df_ints:
            fkernel = kernel_df_C if mp._kernel.lower() == 'c' else kernel_df
        else:
            fkernel = kernel_fftdf
    else:
        log.error('Unknown kernel type')
        raise ValueError

    return fkernel(mp, mo_energy, mo_coeff, eris, with_t2, verbose)

@lib.with_doc(kernel.__doc__)
def kernel_fftdf(mp, mo_energy, mo_coeff, eris=None, with_t2=WITH_T2, verbose=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mp, verbose)
    log.debug('Using FFTDF kernel')

    if eris is None:
        eris = mp.ao2mo(mo_coeff, with_t2)

    if mo_energy is None:
        mo_energy = eris.mo_energy

    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts

    kconserv = mp.khelper.kconserv

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mp, kind="split")

    if with_t2:
        t2 = np.zeros((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=complex)
    else:
        t2 = None

    def get_eia(ki, ka):
        eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
        n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
        eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]
        return eia

    # determine occ batch size
    dsize = 8 if eris.dtype == np.float64 else 16
    mem_avail = mp.max_memory - lib.current_memory()[0]
    # 4*[O]^2*V^2 = mem
    occ_blksize = min(nocc, max(1, int(np.floor((0.7*mem_avail*0.25*1e6/dsize / nvir**2.)**0.5))))
    log.debug('occ blksize for %s loop: %d/%d', mp.__class__.__name__, occ_blksize, nocc)

    cput1 = (logger.process_clock(), logger.perf_counter())

    tspans = np.zeros((2,2))
    tnames = ['ovov', 'energy']

    emp2_ss = emp2_os = 0.
    for ki in range(nkpts):
        for kj in range(ki+1):
            fac_kikj = 1 if ki==kj else 2
            done = {(ka,kconserv[ki,ka,kj]):False for ka in range(nkpts)}
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]

                if done[(ka,kb)] or done[(kb,ka)]:
                    continue

                eia = get_eia(ki, ka)
                ejb = get_eia(kj, kb)

                ovov_ij = [None] * 2
                for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc,occ_blksize)):
                    for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc,occ_blksize)):
                        TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
                        ovov_ij[0] = eris.get_ovov((ki,ka,kj,kb), (i0,i1), (j0,j1))
                        if ka == kb:
                            fac_swap = 1
                            ovov_ij[1] = ovov_ij[0]
                        else:
                            fac_swap = 2
                            ovov_ij[1] = eris.get_ovov((ki,kb,kj,ka), (i0,i1), (j0,j1))
                        TOCK = np.asarray((logger.process_clock(), logger.perf_counter()))
                        tspans[0] += TOCK - TICK

                        eiajb = lib.direct_sum('ia,jb->iajb', eia[i0:i1], ejb[j0:j1])

                        t2_iajb = np.conj(ovov_ij[0] / eiajb)
                        if with_t2:
                            t2[ki,kj,ka][i0:i1,j0:j1] = t2_iajb.transpose(0,2,1,3)
                            if ki != kj:
                                t2[kj,ki,kb][j0:j1,i0:i1] = t2_iajb.transpose(2,0,3,1)

                        edi = einsum('iajb,iajb', t2_iajb, ovov_ij[0]).real * fac_kikj
                        exi = -einsum('iajb,ibja', t2_iajb, ovov_ij[1]).real * fac_swap * fac_kikj
                        emp2_ss += edi + exi
                        emp2_os += edi

                        t2_iajb = None

                        if ka != kb:
                            t2_ibja = np.conj(ovov_ij[1] / eiajb.transpose(0,3,2,1))
                            if with_t2:
                                t2[ki,kj,kb][i0:i1,j0:j1] = t2_ibja.transpose(0,2,1,3)
                                if ki != kj:
                                    t2[kj,ki,ka][j0:j1,i0:i1] = t2_ibja.transpose(2,0,3,1)

                            edi = einsum('iajb,iajb', t2_ibja, ovov_ij[1]).real * fac_kikj
                            emp2_ss += edi
                            emp2_os += edi

                            t2_ibja = None

                        eiajb = None

                        TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
                        tspans[1] += TICK - TOCK

                ovov_ij = None
                done[(ka,kb)] = done[(kb,ka)] = True

        cput1 = log.timer_debug1('ki = %d' % ki, *cput1)

    log.debug('')
    for tspan,tname in zip(tspans,tnames):
        log.debug(f'    CPU time for {tname:10s} {tspan[0]:9.2f} sec, wall time {tspan[1]:9.2f} sec')
    log.debug('')

    log.timer(mp.__class__.__name__, *cput0)

    emp2_ss /= nkpts
    emp2_os /= nkpts
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2, t2

@lib.with_doc(kernel.__doc__)
def kernel_df(mp, mo_energy, mo_coeff, eris=None, with_t2=WITH_T2, verbose=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mp, verbose)
    log.debug('Using DF-Python kernel')

    if eris is None:
        eris = mp.ao2mo(mo_coeff, with_t2)

    if mo_energy is None:
        mo_energy = eris.mo_energy

    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts

    kconserv = mp.khelper.kconserv

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mp, kind="split")

    if with_t2:
        t2 = np.zeros((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=complex)
    else:
        t2 = None

    def get_eia(ki, ka):
        eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
        n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
        eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]
        return eia

    # determine occ batch size
    naux = mp._scf.with_df.get_naoaux()
    dsize = 8 if eris.dtype == np.float64 else 16
    mem_avail = mp.max_memory - lib.current_memory()[0]
    # 4*[O]^2*V^2 + 4*[O]XV = mem
    occ_blksize = min(nocc, max(1, int(np.floor(((naux**2+0.8*mem_avail*0.25*1e6/dsize)**0.5 -
                                                naux) / (2*nvir)))))
    log.debug('occ blksize for %s loop: %d/%d', mp.__class__.__name__, occ_blksize, nocc)

    cput1 = (logger.process_clock(), logger.perf_counter())

    tspans = np.zeros((3,2))
    tnames = ['load', 'ovov', 'energy']

    emp2_ss = emp2_os = 0.
    for ki in range(nkpts):
        for kj in range(ki+1):
            fac_kikj = 1 if ki==kj else 2
            done = {(ka,kconserv[ki,ka,kj]):False for ka in range(nkpts)}
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]

                if done[(ka,kb)] or done[(kb,ka)]:
                    continue

                eia = get_eia(ki, ka)
                ejb = get_eia(kj, kb)

                ovov_ij = [None] * 2
                for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc,occ_blksize)):
                    TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
                    iaL = eris.get_ovL((ki,ka), (i0,i1))
                    if ka == kb:
                        ibL = iaL
                    else:
                        ibL = eris.get_ovL((ki,kb), (i0,i1))
                    TOCK = np.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[0] += TOCK - TICK
                    for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc,occ_blksize)):
                        TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
                        if ka == kb:
                            fac_swap = 1
                            if ki == kj and ibatch == jbatch:
                                jbL = iaL
                            else:
                                jbL = eris.get_ovL((kj,kb), (j0,j1))
                            TOCK = np.asarray((logger.process_clock(), logger.perf_counter()))
                            tspans[0] += TOCK - TICK
                            ovov_ij[0] = einsum('iaL,jbL->iajb', iaL, jbL) / nkpts
                            ovov_ij[1] = ovov_ij[0]
                            TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
                            tspans[1] += TICK - TOCK
                        else:
                            fac_swap = 2
                            jbL = eris.get_ovL((kj,kb), (j0,j1))
                            jaL = eris.get_ovL((kj,ka), (j0,j1))
                            TOCK = np.asarray((logger.process_clock(), logger.perf_counter()))
                            tspans[0] += TOCK - TICK
                            ovov_ij[0] = einsum('iaL,jbL->iajb', iaL, jbL) / nkpts
                            ovov_ij[1] = einsum('iaL,jbL->iajb', ibL, jaL) / nkpts
                            TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
                            tspans[1] += TICK - TOCK
                        jaL = jbL = None

                        eiajb = lib.direct_sum('ia,jb->iajb', eia[i0:i1], ejb[j0:j1])

                        t2_iajb = np.conj(ovov_ij[0] / eiajb)
                        if with_t2:
                            t2[ki,kj,ka][i0:i1,j0:j1] = t2_iajb.transpose(0,2,1,3)
                            if ki != kj:
                                t2[kj,ki,kb][j0:j1,i0:i1] = t2_iajb.transpose(2,0,3,1)

                        edi = einsum('iajb,iajb', t2_iajb, ovov_ij[0]).real * fac_kikj
                        exi = -einsum('iajb,ibja', t2_iajb, ovov_ij[1]).real * fac_swap * fac_kikj
                        emp2_ss += edi + exi
                        emp2_os += edi

                        t2_iajb = None

                        if ka != kb:
                            t2_ibja = np.conj(ovov_ij[1] / eiajb.transpose(0,3,2,1))
                            if with_t2:
                                t2[ki,kj,kb][i0:i1,j0:j1] = t2_ibja.transpose(0,2,1,3)
                                if ki != kj:
                                    t2[kj,ki,ka][j0:j1,i0:i1] = t2_ibja.transpose(2,0,3,1)

                            edi = einsum('iajb,iajb', t2_ibja, ovov_ij[1]).real * fac_kikj
                            emp2_ss += edi
                            emp2_os += edi

                            t2_ibja = None

                        eiajb = None

                        TOCK = np.asarray((logger.process_clock(), logger.perf_counter()))
                        tspans[2] += TOCK - TICK

                    iaL = ibL = None

                ovov_ij = None
                done[(ka,kb)] = done[(kb,ka)] = True

        cput1 = log.timer_debug1('ki = %d' % ki, *cput1)

    log.debug('')
    for tspan,tname in zip(tspans,tnames):
        log.debug(f'    CPU time for {tname:10s} {tspan[0]:9.2f} sec, wall time {tspan[1]:9.2f} sec')
    log.debug('')

    log.timer(mp.__class__.__name__, *cput0)

    emp2_ss /= nkpts
    emp2_os /= nkpts
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2, t2

@lib.with_doc(kernel.__doc__)
def kernel_df_C(mp, mo_energy, mo_coeff, eris=None, with_t2=WITH_T2, verbose=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mp, verbose)
    log.debug('Using DF-C kernel')

    if eris is None:
        eris = mp.ao2mo(mo_coeff, with_t2)

    if mo_energy is None:
        mo_energy = eris.mo_energy

    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts

    kconserv = mp.khelper.kconserv

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mp, kind="split")

    if with_t2:
        log.error('DF-C kernel does not support with_t2 = True. '
                  'Run with mmp.kernel(with_t2=False) or '
                  'use DF-Python kernel by mmp._kernel = \'py\'.')
        raise NotImplementedError
        t2 = np.zeros((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=complex)
    else:
        t2 = None

    drv = libmp.KMP2_contract_drv

    def get_eij(ki, kj):
        eij = -LARGE_DENOM * np.ones((nocc, nocc), dtype=mo_energy[0].dtype)
        n0_ovp_ij = np.ix_(nonzero_opadding[ki], nonzero_opadding[kj])
        eij[n0_ovp_ij] = (mo_e_o[ki][:,None] + mo_e_o[kj])[n0_ovp_ij]
        return eij
    def get_eab(ka, kb):
        eab = LARGE_DENOM * np.ones((nvir, nvir), dtype=mo_energy[0].dtype)
        n0_ovp_ab = np.ix_(nonzero_vpadding[ka], nonzero_vpadding[kb])
        eab[n0_ovp_ab] = (mo_e_v[ka][:,None] + mo_e_v[kb])[n0_ovp_ab]
        return eab

    # determine occ batch size
    naux = mp._scf.with_df.get_naoaux()
    dsize = 8 if eris.dtype == np.float64 else 16
    mem_avail = mp.max_memory - lib.current_memory()[0]
    # 4*[O]^2*V^2 + 4*[O]XV = mem
    occ_blksize = min(nocc, max(1, int(np.floor(((naux**2+0.8*mem_avail*0.25*1e6/dsize)**0.5 -
                                                naux) / (2*nvir)))))
    log.debug('occ blksize for %s loop: %d/%d', mp.__class__.__name__, occ_blksize, nocc)

    cput1 = (logger.process_clock(), logger.perf_counter())

    tspans = np.zeros((2,2))
    tnames = ['load', 'contract']

    emp2_ss = emp2_os = 0
    for ki in range(nkpts):
        for kj in range(ki+1):
            fac_kikj = 1 if ki==kj else 2
            moeoo = get_eij(ki,kj)
            done = {(ka,kconserv[ki,ka,kj]):False for ka in range(nkpts)}
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]

                if done[(ka,kb)] or done[(kb,ka)]:
                    continue

                moevv = lib.asarray(get_eab(ka,kb).reshape(-1), order='C')

                for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc,occ_blksize)):
                    nocci = i1-i0
                    TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
                    iaLR, iaLI = eris.get_ovL((ki,ka), (i0,i1), True)
                    if ka == kb:
                        ibLR, ibLI = iaLR, iaLI
                    else:
                        ibLR, ibLI = eris.get_ovL((ki,kb), (i0,i1), True)
                    naux = iaLR.shape[-1]
                    for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc,occ_blksize)):
                        noccj = j1-j0
                        if ka == kb:
                            if ki == kj and ibatch == jbatch:
                                jbLR, jbLI = iaLR, iaLI
                            else:
                                jbLR, jbLI = eris.get_ovL((kj,kb), (j0,j1), True)
                            jaLR, jaLI = jbLR, jbLI
                        else:
                            jbLR, jbLI = eris.get_ovL((kj,kb), (j0,j1), True)
                            jaLR, jaLI = eris.get_ovL((kj,ka), (j0,j1), True)
                        TOCK = np.asarray((logger.process_clock(), logger.perf_counter()))
                        tspans[0] += TOCK - TICK

                        ed = np.zeros(1, dtype=np.float64)
                        ex = np.zeros(1, dtype=np.float64)
                        drv(
                            ed.ctypes.data_as(ctypes.c_void_p),
                            ex.ctypes.data_as(ctypes.c_void_p),
                            iaLR.ctypes.data_as(ctypes.c_void_p),
                            iaLI.ctypes.data_as(ctypes.c_void_p),
                            ibLR.ctypes.data_as(ctypes.c_void_p),
                            ibLI.ctypes.data_as(ctypes.c_void_p),
                            jbLR.ctypes.data_as(ctypes.c_void_p),
                            jbLI.ctypes.data_as(ctypes.c_void_p),
                            jaLR.ctypes.data_as(ctypes.c_void_p),
                            jaLI.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(ki), ctypes.c_int(kj),
                            ctypes.c_int(ka), ctypes.c_int(kb),
                            ctypes.c_int(i0), ctypes.c_int(j0),
                            ctypes.c_int(nocci), ctypes.c_int(noccj),
                            ctypes.c_int(nvir), ctypes.c_int(naux),
                            lib.asarray(moeoo[i0:i1,j0:j1],
                                        order='C').ctypes.data_as(ctypes.c_void_p),
                            moevv.ctypes.data_as(ctypes.c_void_p),
                        )
                        ed *= fac_kikj / nkpts**2
                        ex *= fac_kikj / nkpts**2

                        emp2_ss += ed + ex
                        emp2_os += ed

                        TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
                        tspans[1] += TICK - TOCK

                        jaLR = jaLI = jbLR = jbLI = None

                    iaLR = iaLI = ibLR = ibLI = None

                done[(ka,kb)] = done[(kb,ka)] = True

        cput1 = log.timer_debug1('ki = %d' % ki, *cput1)

    log.debug('')
    for tspan,tname in zip(tspans,tnames):
        log.debug(f'    CPU time for {tname:10s} {tspan[0]:9.2f} sec, wall time {tspan[1]:9.2f} sec')
    log.debug('')

    log.timer(mp.__class__.__name__, *cput0)

    emp2_ss /= nkpts
    emp2_os /= nkpts
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2, t2


def _iterative_kernel(self, eris):
    raise NotImplementedError


def _padding_k_idx(nmo, nocc, kind="split"):
    """A convention used for padding vectors, matrices and tensors in case when occupation numbers depend on the
    k-point index.
    Args:
        nmo (Iterable): k-dependent orbital number;
        nocc (Iterable): k-dependent occupation numbers;
        kind (str): either "split" (occupied and virtual spaces are split) or "joint" (occupied and virtual spaces are
        the joint;

    Returns:
        Two lists corresponding to the occupied and virtual spaces for kind="split". Each list contains integer arrays
        with indexes pointing to actual non-zero entries in the padded vector/matrix/tensor. If kind="joint", a single
        list of arrays is returned corresponding to the entire MO space.
    """
    if kind not in ("split", "joint"):
        raise ValueError("The 'kind' argument must be one of 'split', 'joint'")

    if kind == "split":
        indexes_o = []
        indexes_v = []
    else:
        indexes = []

    nocc = np.array(nocc)
    nmo = np.array(nmo)
    nvirt = nmo - nocc
    dense_o = np.amax(nocc)
    dense_v = np.amax(nvirt)
    dense_nmo = dense_o + dense_v

    for k_o, k_nmo in zip(nocc, nmo):
        k_v = k_nmo - k_o
        if kind == "split":
            indexes_o.append(np.arange(k_o))
            indexes_v.append(np.arange(dense_v - k_v, dense_v))
        else:
            indexes.append(np.concatenate((
                np.arange(k_o),
                np.arange(dense_nmo - k_v, dense_nmo),
            )))

    if kind == "split":
        return indexes_o, indexes_v

    else:
        return indexes


def padding_k_idx(mp, kind="split"):
    """A convention used for padding vectors, matrices and tensors in case when occupation numbers depend on the
    k-point index.

    This implementation stores k-dependent Fock and other matrix in dense arrays with additional dimensions
    corresponding to k-point indexes. In case when the occupation numbers depend on the k-point index (i.e. a metal) or
    when some k-points have more Bloch basis functions than others the corresponding data structure has to be padded
    with entries that are not used (fictitious occupied and virtual degrees of freedom). Current convention stores these
    states at the Fermi level as shown in the following example.

    +----+--------+--------+--------+
    |    |  k=0   |  k=1   |  k=2   |
    |    +--------+--------+--------+
    |    | nocc=2 | nocc=3 | nocc=2 |
    |    | nvir=4 | nvir=3 | nvir=3 |
    +====+========+========+========+
    | v3 |  k0v3  |  k1v2  |  k2v2  |
    +----+--------+--------+--------+
    | v2 |  k0v2  |  k1v1  |  k2v1  |
    +----+--------+--------+--------+
    | v1 |  k0v1  |  k1v0  |  k2v0  |
    +----+--------+--------+--------+
    | v0 |  k0v0  |        |        |
    +====+========+========+========+
    |          Fermi level          |
    +====+========+========+========+
    | o2 |        |  k1o2  |        |
    +----+--------+--------+--------+
    | o1 |  k0o1  |  k1o1  |  k2o1  |
    +----+--------+--------+--------+
    | o0 |  k0o0  |  k1o0  |  k2o0  |
    +----+--------+--------+--------+

    In the above example, `get_nmo(mp, per_kpoint=True) == (6, 6, 5)`, `get_nocc(mp, per_kpoint) == (2, 3, 2)`. The
    resulting dense `get_nmo(mp) == 7` and `get_nocc(mp) == 3` correspond to padded dimensions. This function will
    return the following indexes corresponding to the filled entries of the above table:

    >>> padding_k_idx(mp, kind="split")
    ([(0, 1), (0, 1, 2), (0, 1)], [(0, 1, 2, 3), (1, 2, 3), (1, 2, 3)])

    >>> padding_k_idx(mp, kind="joint")
    [(0, 1, 3, 4, 5, 6), (0, 1, 2, 4, 5, 6), (0, 1, 4, 5, 6)]

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        kind (str): either "split" (occupied and virtual spaces are split) or "joint" (occupied and virtual spaces are
        the joint;

    Returns:
        Two lists corresponding to the occupied and virtual spaces for kind="split". Each list contains integer arrays
        with indexes pointing to actual non-zero entries in the padded vector/matrix/tensor. If kind="joint", a single
        list of arrays is returned corresponding to the entire MO space.
    """
    return _padding_k_idx(mp.get_nmo(per_kpoint=True), mp.get_nocc(per_kpoint=True), kind=kind)


def padded_mo_energy(mp, mo_energy):
    """
    Pads energies of active MOs.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        mo_energy (ndarray): original non-padded molecular energies;

    Returns:
        Padded molecular energies.
    """
    frozen_mask = get_frozen_mask(mp)
    padding_convention = padding_k_idx(mp, kind="joint")
    nkpts = mp.nkpts

    result = np.zeros((nkpts, mp.nmo), dtype=mo_energy[0].dtype)
    for k in range(nkpts):
        result[np.ix_([k], padding_convention[k])] = mo_energy[k][frozen_mask[k]]

    return result


def padded_mo_coeff(mp, mo_coeff):
    """
    Pads coefficients of active MOs.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        mo_coeff (ndarray): original non-padded molecular coefficients;

    Returns:
        Padded molecular coefficients.
    """
    frozen_mask = get_frozen_mask(mp)
    padding_convention = padding_k_idx(mp, kind="joint")
    nkpts = mp.nkpts

    result = np.zeros((nkpts, mo_coeff[0].shape[0], mp.nmo), dtype=mo_coeff[0].dtype)
    for k in range(nkpts):
        result[np.ix_([k], np.arange(result.shape[1]), padding_convention[k])] = mo_coeff[k][:, frozen_mask[k]]

    return result


def _frozen_sanity_check(frozen, mo_occ, kpt_idx):
    '''Performs a few sanity checks on the frozen array and mo_occ.

    Specific tests include checking for duplicates within the frozen array.

    Args:
        frozen (array_like of int): The orbital indices that will be frozen.
        mo_occ (:obj:`ndarray` of int): The occupuation number for each orbital
            resulting from a mean-field-like calculation.
        kpt_idx (int): The k-point that `mo_occ` and `frozen` belong to.

    '''
    frozen = np.array(frozen)
    nocc = np.count_nonzero(mo_occ > 0)

    assert nocc, 'No occupied orbitals?\n\nnocc = %s\nmo_occ = %s' % (nocc, mo_occ)
    all_frozen_unique = (len(frozen) - len(np.unique(frozen))) == 0
    if not all_frozen_unique:
        raise RuntimeError('Frozen orbital list contains duplicates!\n\nkpt_idx %s\n'
                           'frozen %s' % (kpt_idx, frozen))
    if len(frozen) > 0 and np.max(frozen) > len(mo_occ) - 1:
        raise RuntimeError('Freezing orbital not in MO list!\n\nkpt_idx %s\n'
                           'frozen %s\nmax orbital idx %s' % (kpt_idx, frozen, len(mo_occ) - 1))


def get_nocc(mp, per_kpoint=False):
    '''Number of occupied orbitals for k-point calculations.

    Number of occupied orbitals for use in a calculation with k-points, taking into
    account frozen orbitals.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        per_kpoint (bool, optional): True returns the number of occupied
            orbitals at each k-point.  False gives the max of this list.

    Returns:
        nocc (int, list of int): Number of occupied orbitals. For return type, see description of arg
            `per_kpoint`.

    '''
    for i, moocc in enumerate(mp.mo_occ):
        if np.any(moocc % 1 != 0):
            raise RuntimeError("Fractional occupation numbers encountered @ kp={:d}: {}. This may have been caused by "
                               "smearing of occupation numbers in the mean-field calculation. If so, consider "
                               "executing mf.smearing_method = False; mf.mo_occ = mf.get_occ() prior to calling "
                               "this".format(i, moocc))
    if mp._nocc is not None:
        return mp._nocc
    elif mp.frozen is None:
        nocc = [np.count_nonzero(mp.mo_occ[ikpt]) for ikpt in range(mp.nkpts)]
    elif isinstance(mp.frozen, (int, np.integer)):
        nocc = [(np.count_nonzero(mp.mo_occ[ikpt]) - mp.frozen) for ikpt in range(mp.nkpts)]
    elif isinstance(mp.frozen[0], (int, np.integer)):
        [_frozen_sanity_check(mp.frozen, mp.mo_occ[ikpt], ikpt) for ikpt in range(mp.nkpts)]
        nocc = []
        for ikpt in range(mp.nkpts):
            max_occ_idx = np.max(np.where(mp.mo_occ[ikpt] > 0))
            frozen_nocc = np.sum(np.array(mp.frozen) <= max_occ_idx)
            nocc.append(np.count_nonzero(mp.mo_occ[ikpt]) - frozen_nocc)
    elif isinstance(mp.frozen[0], (list, np.ndarray)):
        nkpts = len(mp.frozen)
        if nkpts != mp.nkpts:
            raise RuntimeError('Frozen list has a different number of k-points (length) than passed in mean-field/'
                               'correlated calculation.  \n\nCalculation nkpts = %d, frozen list = %s '
                               '(length = %d)' % (mp.nkpts, mp.frozen, nkpts))
        [_frozen_sanity_check(frozen, mo_occ, ikpt) for ikpt, frozen, mo_occ in zip(range(nkpts), mp.frozen, mp.mo_occ)]

        nocc = []
        for ikpt, frozen in enumerate(mp.frozen):
            max_occ_idx = np.max(np.where(mp.mo_occ[ikpt] > 0))
            frozen_nocc = np.sum(np.array(frozen) <= max_occ_idx)
            nocc.append(np.count_nonzero(mp.mo_occ[ikpt]) - frozen_nocc)
    else:
        raise NotImplementedError

    assert any(np.array(nocc) > 0), ('Must have occupied orbitals! \n\nnocc %s\nfrozen %s\nmo_occ %s' %
           (nocc, mp.frozen, mp.mo_occ))

    if not per_kpoint:
        nocc = np.amax(nocc)

    return nocc


def get_nmo(mp, per_kpoint=False):
    '''Number of orbitals for k-point calculations.

    Number of orbitals for use in a calculation with k-points, taking into account
    frozen orbitals.

    Note:
        If `per_kpoint` is False, then the number of orbitals here is equal to max(nocc) + max(nvir),
        where each max is done over all k-points.  Otherwise the number of orbitals is returned
        as a list of number of orbitals at each k-point.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        per_kpoint (bool, optional): True returns the number of orbitals at each k-point.
            For a description of False, see Note.

    Returns:
        nmo (int, list of int): Number of orbitals. For return type, see description of arg
            `per_kpoint`.

    '''
    if mp._nmo is not None:
        return mp._nmo

    if mp.frozen is None:
        nmo = [len(mp.mo_occ[ikpt]) for ikpt in range(mp.nkpts)]
    elif isinstance(mp.frozen, (int, np.integer)):
        nmo = [len(mp.mo_occ[ikpt]) - mp.frozen for ikpt in range(mp.nkpts)]
    elif isinstance(mp.frozen[0], (int, np.integer)):
        [_frozen_sanity_check(mp.frozen, mp.mo_occ[ikpt], ikpt) for ikpt in range(mp.nkpts)]
        nmo = [len(mp.mo_occ[ikpt]) - len(mp.frozen) for ikpt in range(mp.nkpts)]
    elif isinstance(mp.frozen, (list, np.ndarray)):
        nkpts = len(mp.frozen)
        if nkpts != mp.nkpts:
            raise RuntimeError('Frozen list has a different number of k-points (length) than passed in mean-field/'
                               'correlated calculation.  \n\nCalculation nkpts = %d, frozen list = %s '
                               '(length = %d)' % (mp.nkpts, mp.frozen, nkpts))
        [_frozen_sanity_check(fro, mo_occ, ikpt) for ikpt, fro, mo_occ in zip(range(nkpts), mp.frozen, mp.mo_occ)]

        nmo = [len(mp.mo_occ[ikpt]) - len(mp.frozen[ikpt]) for ikpt in range(nkpts)]
    else:
        raise NotImplementedError

    assert all(np.array(nmo) > 0), ('Must have a positive number of orbitals!\n\nnmo %s\nfrozen %s\nmo_occ %s' %
           (nmo, mp.frozen, mp.mo_occ))

    if not per_kpoint:
        # Depending on whether there are more occupied bands, we want to make sure that
        # nmo has enough room for max(nocc) + max(nvir) number of orbitals for occupied
        # and virtual space
        nocc = mp.get_nocc(per_kpoint=True)
        nmo = np.max(nocc) + np.max(np.array(nmo) - np.array(nocc))

    return nmo


def get_frozen_mask(mp):
    '''Boolean mask for orbitals in k-point post-HF method.

    Creates a boolean mask to remove frozen orbitals and keep other orbitals for post-HF
    calculations.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.

    Returns:
        moidx (list of :obj:`ndarray` of `bool`): Boolean mask of orbitals to include.

    '''
    moidx = [np.ones(x.size, dtype=bool) for x in mp.mo_occ]
    if mp.frozen is None:
        pass
    elif isinstance(mp.frozen, (int, np.integer)):
        for idx in moidx:
            idx[:mp.frozen] = False
    elif isinstance(mp.frozen[0], (int, np.integer)):
        frozen = list(mp.frozen)
        for idx in moidx:
            idx[frozen] = False
    elif isinstance(mp.frozen[0], (list, np.ndarray)):
        nkpts = len(mp.frozen)
        if nkpts != mp.nkpts:
            raise RuntimeError('Frozen list has a different number of k-points (length) than passed in mean-field/'
                               'correlated calculation.  \n\nCalculation nkpts = %d, frozen list = %s '
                               '(length = %d)' % (mp.nkpts, mp.frozen, nkpts))
        [_frozen_sanity_check(fro, mo_occ, ikpt) for ikpt, fro, mo_occ in zip(range(nkpts), mp.frozen, mp.mo_occ)]
        for ikpt, kpt_occ in enumerate(moidx):
            kpt_occ[mp.frozen[ikpt]] = False
    else:
        raise NotImplementedError

    return moidx


def _add_padding(mp, mo_coeff, mo_energy):
    nmo = mp.nmo

    # Check if these are padded mo coefficients and energies and/or if some orbitals are frozen.
    if (mp.frozen is not None) or (not np.all([x.shape[1] == nmo for x in mo_coeff])):
        mo_coeff = padded_mo_coeff(mp, mo_coeff)

    if (mp.frozen is not None) or (not np.all([x.shape[0] == nmo for x in mo_energy])):
        mo_energy = padded_mo_energy(mp, mo_energy)
    return mo_coeff, mo_energy


def make_rdm1(mp, t2=None, kind="compact"):
    r"""
    Spin-traced one-particle density matrix in the MO basis representation.
    The occupied-virtual orbital response is not included.

    dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)

    Args:
        mp (KMP2): a KMP2 kernel object;
        t2 (ndarray): a t2 MP2 tensor;
        kind (str): either 'compact' or 'padded' - defines behavior for k-dependent MO basis sizes;

    Returns:
        A k-dependent single-particle density matrix.
    """
    if kind not in ("compact", "padded"):
        raise ValueError("The 'kind' argument should be either 'compact' or 'padded'")
    d_imds = _gamma1_intermediates(mp, t2=t2)
    result = []
    padding_idxs = padding_k_idx(mp, kind="joint")
    for (oo, vv), idxs in zip(zip(*d_imds), padding_idxs):
        oo += np.eye(*oo.shape)
        d = block_diag(oo, vv)
        d += d.conj().T
        if kind == "padded":
            result.append(d)
        else:
            result.append(d[np.ix_(idxs, idxs)])
    return result


def make_rdm2(mp, t2=None, kind="compact"):
    r'''
    Spin-traced two-particle density matrix in MO basis

    .. math::

        dm2[p,q,r,s] = \sum_{\sigma,\tau} <p_\sigma^\dagger r_\tau^\dagger s_\tau q_\sigma>

    Note the contraction between ERIs (in Chemist's notation) and rdm2 is
    E = einsum('pqrs,pqrs', eri, rdm2)
    '''
    if kind not in ("compact", "padded"):
        raise ValueError("The 'kind' argument should be either 'compact' or 'padded'")
    if t2 is None: t2 = mp.t2
    dm1 = mp.make_rdm1(t2, "padded")
    nmo = mp.nmo
    nocc = mp.nocc
    nkpts = mp.nkpts
    dtype = t2.dtype

    dm2 = np.zeros((nkpts,nkpts,nkpts,nmo,nmo,nmo,nmo),dtype=dtype)
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = mp.khelper.kconserv[ki, ka, kj]
                dovov = t2[ki, kj, ka].transpose(0,2,1,3) * 2 - t2[kj, ki, ka].transpose(1,2,0,3)
                dovov *= 2
                dm2[ki,ka,kj,:nocc,nocc:,:nocc,nocc:] = dovov
                dm2[ka,ki,kb,nocc:,:nocc,nocc:,:nocc] = dovov.transpose(1,0,3,2).conj()

    occidx = padding_k_idx(mp, kind="split")[0]
    for ki in range(nkpts):
        for i in occidx[ki]:
            dm1[ki][i,i] -= 2

    for ki in range(nkpts):
        for kp in range(nkpts):
            for i in occidx[ki]:
                dm2[ki,ki,kp,i,i,:,:] += dm1[kp].T * 2
                dm2[kp,kp,ki,:,:,i,i] += dm1[kp].T * 2
                dm2[kp,ki,ki,:,i,i,:] -= dm1[kp].T
                dm2[ki,kp,kp,i,:,:,i] -= dm1[kp]

    for ki in range(nkpts):
        for kj in range(nkpts):
            for i in occidx[ki]:
                for j in occidx[kj]:
                    dm2[ki,ki,kj,i,i,j,j] += 4
                    dm2[ki,kj,kj,i,j,j,i] -= 2

    if kind == "padded":
        return dm2
    else:
        idx = padding_k_idx(mp, kind="joint")
        result = np.ndarray((nkpts,nkpts,nkpts), dtype=object)
        for kp in range(nkpts):
            for kq in range(nkpts):
                for kr in range(nkpts):
                    ks = mp.khelper.kconserv[kp, kq, kr]
                    result[kp,kq,kr] = dm2[kp,kq,kr][np.ix_(idx[kp],idx[kq],idx[kr],idx[ks])]
        return result


def _gamma1_intermediates(mp, t2=None):
    # Memory optimization should be here
    if t2 is None:
        t2 = mp.t2
    if t2 is None:
        raise NotImplementedError("Run kmp2.kernel with `with_t2=True`")
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts
    dtype = t2.dtype

    dm1occ = np.zeros((nkpts, nocc, nocc), dtype=dtype)
    dm1vir = np.zeros((nkpts, nvir, nvir), dtype=dtype)

    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = mp.khelper.kconserv[ki, ka, kj]

                dm1vir[kb] += einsum('ijax,ijay->yx', t2[ki][kj][ka].conj(), t2[ki][kj][ka]) * 2 -\
                              einsum('ijax,ijya->yx', t2[ki][kj][ka].conj(), t2[ki][kj][kb])
                dm1occ[kj] += einsum('ixab,iyab->xy', t2[ki][kj][ka].conj(), t2[ki][kj][ka]) * 2 -\
                              einsum('ixab,iyba->xy', t2[ki][kj][ka].conj(), t2[ki][kj][kb])
    return -dm1occ, dm1vir


class KMP2(mp2.MP2):

    _kernel = getattr(__config__, 'pbc_mp_KMP2_kernel', None)

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):

        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ is None: mo_occ = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen
        if isinstance(self._scf.with_df, df.GDF):
            self.with_df_ints = True
        else:
            self.with_df_ints = False

        # for GDF
        self._ovL = None
        self._ovL_to_save = None

##################################################
# don't modify the following attributes, they are not input options
        self.kpts = mf.kpts
        if isinstance(self.kpts, libkpts.KPoints):
            self.nkpts = self.kpts.nkpts
            self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts.kpts)
            #padding has to be after transformation
            self.mo_energy = self.kpts.transform_mo_energy(mf.mo_energy)
            self.mo_coeff = self.kpts.transform_mo_coeff(mo_coeff)
            self.mo_occ = self.kpts.transform_mo_occ(mo_occ)
        else:
            self.nkpts = len(self.kpts)
            self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
            self.mo_energy = mf.mo_energy
            self.mo_coeff = mo_coeff
            self.mo_occ = mo_occ
        self._nocc = None
        self._nmo = None
        self.e_hf = None
        self.e_corr = None
        self.e_corr_ss = None
        self.e_corr_os = None
        self.t2 = None
        self._keys = set(self.__dict__.keys())

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    make_rdm1 = make_rdm1
    make_rdm2 = make_rdm2

    def dump_flags(self):
        logger.info(self, "")
        logger.info(self, "******** %s ********", self.__class__)
        logger.info(self, "nkpts = %d", self.nkpts)
        logger.info(self, "nocc = %s", self.nocc)
        logger.info(self, "nmo = %s", self.nmo)
        logger.info(self, "with_df_ints = %s", self.with_df_ints)
        logger.info(self, "_kernel = %s", self._kernel)

        if self.frozen is not None:
            logger.info(self, "frozen orbitals = %s", self.frozen)
        logger.info(
            self,
            "max_memory %d MB (current use %d MB)",
            self.max_memory,
            lib.current_memory()[0],
        )
        return self

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        '''
        Args:
            with_t2 : bool
                Whether to generate and hold t2 amplitudes in memory.
        '''
        if self.verbose >= logger.WARN:
            self.check_sanity()

        self.dump_flags()

        self.e_hf = self.get_e_hf(mo_coeff=mo_coeff)

        if eris is None:
            eris = self.ao2mo(mo_coeff, with_t2)

        if self._scf.converged:
            self.e_corr, self.t2 = self.init_amps(mo_energy, mo_coeff, eris, with_t2)
        else:
            self.converged, self.e_corr, self.t2 = _iterative_kernel(self, eris)

        self.e_corr_ss = getattr(self.e_corr, 'e_corr_ss', 0)
        self.e_corr_os = getattr(self.e_corr, 'e_corr_os', 0)
        self.e_corr = float(self.e_corr)

        self._finalize()

        return self.e_corr, self.t2

    def ao2mo(self, mo_coeff=None, with_t2=WITH_T2):
        return _make_df_eris(self, mo_coeff, with_t2, verbose=self.verbose)

    # For non-canonical MP2
    # energy = energy
    # update_amps = update_amps
    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)


def _mem_usage(nocc, nvir, nkpts, naux, dsize, with_t2=WITH_T2):
    '''
        basic   = t2 + 4*v**2 + 4*x*v
        incore  = kLov + basic
        outcore = basic
    '''
    # 4 [o][o]vv for ovov(ka,kb), ovov(kb,ka), t2, eiajb + 4 L[o]v
    basic = 4*(nvir**2 + naux*nvir)
    if with_t2:
        basic += nkpts**3*(nocc*nvir)**2
    basic *= dsize/1e6
    incore = nkpts**2*naux*nocc*nvir * dsize/1e6 + basic
    outcore = basic
    return incore, outcore, basic

class _ChemistsERIs:
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None
        self.dtype = None

        # for GDF
        self._ovL = None
        self._ovL_to_save = None
        self.ovL = None

        # for FFTDF
        self._get_ovov = None

    def _common_init_(self, mp, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mp.mo_coeff
        if mo_coeff is None:
            raise RuntimeError('mo_coeff, mo_energy are not initialized.\n'
                               'You may need to call mf.kernel() to generate them.')

        self.mo_coeff = padded_mo_coeff(mp, mo_coeff)
        self.mol = mp.mol

        if gamma_point(mp._scf.kpts):
            self.dtype = np.float64
        else:
            self.dtype = np.complex128

        if mo_coeff is mp._scf.mo_coeff and mp._scf.converged:
            # The canonical MP2 from a converged SCF result. Rebuilding fock can be skipped
            self.mo_energy = padded_mo_energy(mp, mp._scf.mo_energy)
            self.fock = [np.diag(moe) for moe in self.mo_energy]
        else:
            dm = mp._scf.make_rdm1(mo_coeff, mp.mo_occ)
            vhf = mp._scf.get_veff(mp.mol, dm)
            fockao = mp._scf.get_fock(vhf=vhf, dm=dm)
            self.fock = [C.conj().T.dot(f).dot(C) for f,C in zip(fockao,mo_coeff)]
            mo_energy = [f.diagonal().real for f in self.fock]

            self.mo_energy = padded_mo_energy(mp, mo_energy)
        return self

    def get_ovov(self, kiajb, i01, j01):
        return self._get_ovov(kiajb, i01, j01)

    def get_ovL(self, kia, i01, RIsep=False):
        ki,ka = kia
        i0,i1 = i01
        ovL = self.ovL
        if isinstance(ovL, np.ndarray):
            if RIsep:
                return (np.asarray(ovL[ki,ka][i0:i1].real, order='C'),
                        np.asarray(ovL[ki,ka][i0:i1].imag, order='C'))
            else:
                return np.asarray(ovL[ki,ka][i0:i1])
        else:
            if RIsep:
                return (np.asarray(ovL[f'{ki},{ka}'][i0:i1].real, order='C'),
                        np.asarray(ovL[f'{ki},{ka}'][i0:i1].imag, order='C'))
            else:
                return np.asarray(ovL[f'{ki},{ka}'][i0:i1])


def _make_df_eris(mymp, mo_coeff=None, with_t2=WITH_T2, verbose=None):
    log = logger.new_logger(mymp, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mymp, mo_coeff)
    eris._ovL = mymp._ovL
    eris._ovL_to_save = mymp._ovL_to_save
    mo_coeff = eris.mo_coeff
    mf = mymp._scf
    kpts = mf.kpts

    if gamma_point(kpts):
        dtype = np.float64
        dsize = 8
    else:
        dtype = np.complex128
        dsize = 16

    # determine incore/outcore
    nocc = mymp.nocc
    nmo = mymp.nmo
    nvir = nmo - nocc
    nkpts = len(kpts)

    with_df_ints = mymp.with_df_ints and isinstance(mymp._scf.with_df, df.GDF)

    if with_df_ints:
        naux = mymp._scf.with_df.get_naoaux()
        mem_incore, mem_outcore, mem_basic = _mem_usage(nocc, nvir, nkpts, naux, dsize, with_t2)
        mem_now = lib.current_memory()[0]
        max_memory = max(0, mymp.max_memory - mem_now)
        if max_memory < mem_basic:
            log.warn('Not enough memory for integral transformation. '
                     'Available mem %s MB, required mem %s MB',
                     max_memory, mem_basic)

        if eris._ovL is not None:
            if isinstance(eris._ovL, np.ndarray):
                eris.ovL = eris._ovL
                log.debug('Incore 3c integrals are found')
            else:
                eris.ovL = h5py.File(eris._ovL, 'r')
                log.debug('Outcore 3c integrals are found %s', eris._ovL)
        else:
            if mymp.mol.incore_anyway or mem_incore < max_memory:
                eris.ovL = np.ndarray((nkpts,nkpts), dtype=object)
                log.debug('Transformed 3c integrals will be saved in memory')
            else:
                if eris._ovL_to_save is None:
                    eris._ovL_to_save = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
                if isinstance(eris._ovL_to_save, str):
                    eris.ovL = h5py.File(eris._ovL_to_save, 'w')
                    log.debug('Transformed 3c integrals will be saved in %s', eris._ovL_to_save)
                else:
                    eris.ovL = h5py.File(eris._ovL_to_save.name, 'w')
                    log.debug('Transformed 3c integrals will be saved in %s', eris._ovL_to_save.name)

            ovL = _init_mp_df_eris(mymp, mo_coeff, nocc, eris.ovL)
    else:
        fao2mo = mymp._scf.with_df.ao2mo

        def get_ovov(kiajb, i01, j01):
            ki,ka,kj,kb = kiajb
            i0, i1 = i01
            j0, j1 = j01
            nocci = i1-i0
            noccj = j1-j0

            orbo_i = mo_coeff[ki][:,i0:i1]
            orbo_j = mo_coeff[kj][:,j0:j1]
            orbv_a = mo_coeff[ka][:,nocc:]
            orbv_b = mo_coeff[kb][:,nocc:]
            ovov = fao2mo((orbo_i,orbv_a,orbo_j,orbv_b),
                          (kpts[ki],kpts[ka],kpts[kj],kpts[kb]),
                          compact=False) / nkpts
            return ovov.reshape(nocci,nvir,noccj,nvir)

        eris._get_ovov = get_ovov

    log.timer('%s ao2mo'%(mymp.__class__.__name__), *time0)
    return eris

def _init_mp_df_eris(mymp, mo_coeff, nocc, ovL=None):
    """Compute 3-center electron repulsion integrals, i.e. (L|ov),
    where `L` denotes DF auxiliary basis functions and `o` and `v` occupied and virtual
    canonical crystalline orbitals. Note that `o` and `v` contain kpt indices `ko` and `kv`,
    and the third kpt index `kL` is determined by the conservation of momentum.

    Arguments:
        mp (KMP2) -- A KMP2 instance

    Returns:
        ovL (np.ndarray) -- 3-center DF ints, with shape (nkpts, nkpts, nocc, nvir, naux)
    """
    from pyscf.ao2mo import _ao2mo

    log = logger.Logger(mymp.stdout, mymp.verbose)

    mydf = mymp._scf.with_df
    if mydf._cderi is None:
        mydf.build()

    cell = mymp._scf.cell
    if cell.dimension == 2:
        # 2D ERIs are not positive definite. The 3-index tensors are stored in
        # two part. One corresponds to the positive part and one corresponds
        # to the negative part. The negative part is not considered in the
        # DF-driven CCSD implementation.
        raise NotImplementedError

    if mo_coeff is None:
        mo_coeff = _add_padding(mp, mymp.mo_coeff, mymp.mo_energy)[0]

    if nocc is None: nocc = mymp.nocc
    nmo = mo_coeff[0].shape[1]
    nvir = nmo - nocc
    nao = cell.nao_nr()
    kpts = mymp.kpts
    nkpts = len(kpts)
    naux_perk = np.zeros((nkpts,nkpts), dtype=int)
    for ki in range(nkpts):
        for kj in range(nkpts):
            kpti_kptj = np.asarray((kpts[ki],kpts[kj]))
            with df._load3c(mydf._cderi, mydf._dataname, kpti_kptj=kpti_kptj) as j3c:
                naux_perk[ki,kj] = j3c.shape[0]
    naux0 = naux_perk.reshape(-1).max()

    if gamma_point(kpts):
        dtype = np.float64
    else:
        dtype = np.complex128
    dtype = np.result_type(dtype, *mo_coeff)
    dsize = 8 if dtype == np.float64 else 16

    if ovL is None:
        ovL = np.empty((nkpts, nkpts), dtype=object)

    tao = []
    ao_loc = None

    def fao2mo(j3c, mo, i0, i1, p0, p1, buf):
        ijslice = (i0,i1,nmo+nocc,nmo*2)

        if dtype == np.double:
            Lpq_ao = np.asarray(j3c[p0:p1].real)
            return _ao2mo.nr_e2(Lpq_ao, mo, ijslice, aosym='s2', out=buf)
        else:
            Lpq_ao = np.asarray(j3c[p0:p1])
            if Lpq_ao[0].size != nao**2:  # aosym = 's2'
                Lpq_ao = lib.unpack_tril(Lpq_ao).astype(np.complex128)
            return _ao2mo.r_e2(Lpq_ao, mo, ijslice, tao, ao_loc, out=buf)

    mem_avail = mymp.max_memory - lib.current_memory()[0]
    if isinstance(ovL, np.ndarray):
        mem_avail -= nkpts**2*naux0*nocc*nvir * dsize/1e6

    if isinstance(ovL, np.ndarray):
        # incore: batching aux (OV + Nao_pair) * [X] = M
        mem_auxblk = (nao**2+nocc*nvir) * dsize/1e6
        aux_blksize = min(naux0, max(1, int(np.floor(mem_avail*0.7 / mem_auxblk))))
        log.debug('aux blksize for incore ao2mo: %d/%d', aux_blksize, naux0)
        buf = np.empty(aux_blksize*nocc*nvir, dtype=dtype)

        for ki in range(nkpts):
            for kj in range(nkpts):
                kpti_kptj = np.asarray((kpts[ki],kpts[kj]))

                mo = np.hstack((mo_coeff[ki], mo_coeff[kj]))
                mo = np.asarray(mo, dtype=dtype, order='F')

                naux = naux_perk[ki,kj]
                ovLij = ovL[ki,kj] = np.empty((nocc,nvir,naux), dtype=dtype)

                with df._load3c(mydf._cderi, mydf._dataname, kpti_kptj=kpti_kptj) as j3c:
                    def process(aux_range):
                        return fao2mo(j3c, mo, 0,nocc, *aux_range, buf)
                    for p0,p1 in lib.prange(0, naux, aux_blksize):
                        out = process((p0,p1))
                        ovLij[:,:,p0:p1] = out.reshape(-1,nocc,nvir).transpose(1,2,0)
                        out = None
                ovLij = None
        buf = None
    else:
        # outcore: batching occ [O]XV and aux ([O]V + Nao_pair)*[X]
        mem_occblk = naux0*nvir * dsize/1e6
        occ_blksize = min(nocc, max(1, int(np.floor(mem_avail*0.6 / mem_occblk))))
        mem_auxblk = (occ_blksize*nvir+nao**2) * dsize/1e6
        aux_blksize = min(naux0, max(1, int(np.floor(mem_avail*0.3 / mem_auxblk))))
        log.debug('occ blksize for outcore ao2mo: %d/%d', occ_blksize, nocc)
        log.debug('aux blksize for outcore ao2mo: %d/%d', aux_blksize, naux0)
        buf = np.empty(naux0*occ_blksize*nvir, dtype=dtype)
        buf2 = np.empty(aux_blksize*occ_blksize*nvir, dtype=dtype)

        for ki in range(nkpts):
            for kj in range(nkpts):
                kpti_kptj = np.asarray((kpts[ki],kpts[kj]))

                mo = np.hstack((mo_coeff[ki], mo_coeff[kj]))
                mo = np.asarray(mo, dtype=dtype, order='F')

                naux = naux_perk[ki,kj]
                ovLij = ovL.create_dataset(f'{ki},{kj}', shape=(nocc,nvir,naux), dtype=dtype)

                for i0,i1 in lib.prange(0, nocc, occ_blksize):
                    nocci = i1-i0
                    OvL = np.ndarray((nocci,nvir,naux), buffer=buf, dtype=dtype)
                    with df._load3c(mydf._cderi, mydf._dataname, kpti_kptj=kpti_kptj) as j3c:
                        def process(aux_range):
                            return fao2mo(j3c, mo, i0,i1, *aux_range, buf2)
                        for p0,p1 in lib.prange(0, naux, aux_blksize):
                            out = process((p0,p1))
                            OvL[:,:,p0:p1] = out.reshape(-1,nocci,nvir).transpose(1,2,0)
                            out = None
                    ovLij[i0:i1] = OvL
                    OvL = None
                ovLij = None
        buf = buf2 = None

    return ovL


KRMP2 = KMP2


from pyscf.pbc import scf
scf.khf.KRHF.MP2 = lib.class_as_method(KRMP2)
scf.kghf.KGHF.MP2 = None
scf.krohf.KROHF.MP2 = None


if __name__ == '__main__':
    import os
    from pyscf.pbc import gto, scf, mp

    atom = '''
    O          0.00000        0.00000        0.11779
    H          0.00000        0.75545       -0.47116
    H          0.00000       -0.75545       -0.47116
    '''
    a = np.eye(3)*3
    basis = 'cc-pvdz'
    pseudo = None
    kmesh = (3,1,1)
    eref = -0.200609535862307

    ''' Uncomment for an example showing different occ for different kpt
    '''
    # a0 = 3.44
    # atom = f'Li 0 0 0; Li {a0*0.5} {a0*0.5} {a0*0.5}'
    # a = np.eye(3) * a0
    # basis = '''
    # Li  S
    # 7.298459    2.818219e-01 -4.209100e-02
    # 2.139580    4.562697e-01 -9.431562e-02
    # 0.670836    3.831236e-01 -1.360431e-01
    # Li  S
    #     0.047832    1.000000e+00
    # Li  P
    #     7.853830    5.386932e-03
    #     2.204330    7.604557e-03
    #     0.513280    1.066386e-01
    # Li  P
    #     0.079100    1.000000e+00
    # Li  D
    #     0.100720    1.000000e+00
    # '''
    # pseudo = 'gth-hf-rev'
    # kmesh = (2,2,1)
    # eref = -0.0763713201505293

    cell = gto.M(atom=atom, a=a, basis=basis, pseudo=pseudo)
    kpts = cell.make_kpts(kmesh)

    mf = scf.KRHF(cell, kpts).density_fit()
    mf.kernel()

    mymp = KMP2(mf, frozen=0).set(verbose=6)
    mymp.kernel(with_t2=False)
    print(mymp.e_corr - eref)

    # test calc mo_energy from given mo_coeff
    mo_coeff = [c.copy() for c in mf.mo_coeff]
    mymp.kernel(mo_coeff=mo_coeff, with_t2=False)
    print(mymp.e_corr - eref)
