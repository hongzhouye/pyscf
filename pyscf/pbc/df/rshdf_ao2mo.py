#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
import time
import numpy as np

from pyscf.pbc import df
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv
from pyscf.pbc.tools import k2gamma
from pyscf import ao2mo
from pyscf import lib
from pyscf import __config__

from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique
from pyscf.pbc.df.df_jk import zdotNC


def get_eri(mydf, kpts=None,
            compact=getattr(__config__, 'pbc_hdf_ao2mo_get_eri', True)):

    t1 = (time.clock(), time.time())
    t2 = (time.clock(), time.time())
    log = lib.logger.Logger(mydf.stdout, mydf.verbose)
    log.info("Computing short-range ERI")
    eri = get_eri_sr(mydf, kpts=kpts, compact=compact)
    t2 = log.timer_debug1('sr eri', *t2)
    eri += get_eri_lr(mydf, kpts=kpts, compact=compact)
    t2 = log.timer_debug1('lr eri', *t2)
    t1 = log.timer_debug1('total eri', *t1)

    return eri

def get_eri_sr(mydf, kpts=None,
               compact=getattr(__config__, 'pbc_hdf_ao2mo_get_eri', True)):
    return df.df_ao2mo.get_eri(mydf, kpts=kpts, compact=compact)

def get_aop_idx(cell_fat, aosym='s2'):
    nbas = cell_fat.nbas
    nao = cell_fat.nao_nr()
    nbas_c, nbas_d = cell_fat._nbas_each_set
    ao_loc = cell_fat.ao_loc_nr()

    c_loc = ao_loc[:nbas_c+1]
    d_loc = ao_loc[nbas_c:nbas+1]
    nao_c = c_loc[-1]
    nao_d = nao - nao_c

    c_rg = np.concatenate([np.arange(c_loc[i],c_loc[i+1])
                          for i in range(nbas_c)])
    d_rg = np.concatenate([np.arange(d_loc[i],d_loc[i+1])
                          for i in range(nbas_d)])

    if aosym == 's2':
        nao_c2 = nao_c*(nao_c+1) // 2
        cc_idx = np.arange(nao_c2)

        A = np.zeros((nao,nao), dtype=int)
        nao2 = nao*(nao+1) // 2
        A[np.tril_indices(nao)] = np.arange(nao2)
        dc_idx = (d_rg[:,None]*nao + c_rg).ravel()
        cd_idx = A.ravel()[dc_idx]

        dd_idx = A[-nao_d:,-nao_d:][np.tril_indices(nao_d)]
    else:
        cc_idx = (c_rg[:,None]*nao + c_rg).ravel()
        cd_idx = np.concatenate([(c_rg[:,None]*nao + d_rg).ravel(),
                                (d_rg[:,None]*nao + c_rg).ravel()])
        dd_idx = (d_rg[:,None]*nao + d_rg).ravel()

    return cc_idx, cd_idx, dd_idx


def fat_orig_loop(cell_fat, cell, aosym='s2', sr_only=False, verbose=0):
    if aosym == 's2':
        return _fat_orig_loop_s2(cell_fat, cell, sr_only, verbose)
    elif aosym == 's1':
        return _fat_orig_loop_s1(cell_fat, cell, sr_only, verbose)
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


def _fat_orig_loop_s2(cell_fat, cell, sr_only=False, verbose=0):

    log = lib.logger.Logger(cell.stdout, verbose)

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

    shlpr_mask = np.ones((nbas,nbas), dtype=int)
    if sr_only:
        shlpr_mask[nbas_c:,nbas_c:] = 0

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


def _fat_orig_loop_s1(cell_fat, cell, sr_only=False, verbose=0):
    log = lib.logger.Logger(sys.stdout, verbose)

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

    shlpr_mask = np.ones((nbas,nbas), dtype=int)
    if sr_only:
        shlpr_mask[nbas_c:,nbas_c:] = 0

    for i in range(nbas):
        i0 = bas_f2o[i]
        for j in range(nbas):
            if not shlpr_mask[i,j]:
                continue

            j0 = bas_f2o[j]
            ap = (ao_rg[i][:,None]*nao + ao_rg[j]).ravel()
            ap0 = (ao_rg0[i0][:,None]*nao0 + ao_rg0[j0]).ravel()

            yield ap, ap0


def get_eri_lr(mydf, kpts=None,
               compact=getattr(__config__, 'pbc_hdf_ao2mo_get_eri', True)):
    """ Long-range part of compact shells, i.e., (cc|cc), (cc|cd), (cd|cd) and their permutational counterparts.

    This function is modified from pyscf.pbc.df.aft_ao2mo.py
    """
    abs_omega = abs(mydf.omega)

    if mydf.use_bvkcell:
        bvk_kmesh = k2gamma.kpts_to_kmesh(mydf.cell, mydf.kpts)
    else:
        bvk_kmesh = None

    # case - no diffuse aos
    if mydf.cell_fat is None:
        with mydf.with_range_coulomb(abs_omega):
            lrdf = df.AFTDF(mydf.cell)
            lrdf.mesh = mydf.mesh_sr
            eri_lr = get_eri_lr_aftdf(lrdf, kpts=kpts, compact=compact,
                                      bvk_kmesh=bvk_kmesh)

        return eri_lr

    t1 = (time.clock(), time.time())
    log = lib.logger.Logger(mydf.stdout, mydf.verbose)

    # case - has diffuse aos
    cell = mydf.cell
    cell_fat = mydf.cell_fat
    nao = cell.nao_nr()
    nao_fat = cell_fat.nao_nr()
    nao_fat2 = nao_fat * nao_fat
    nao_fat2_s2 = nao_fat*(nao_fat+1) // 2
    nbas_c, nbas_d = cell_fat._nbas_each_set

    # first, computing long-range part of non-dd blocks, i.e., (cc|cc), (cc|cd), (cd|cd) and their permutational counterparts
    with cell_fat.with_range_coulomb(abs_omega):
        lrdf = df.AFTDF(cell_fat)
        lrdf.mesh = mydf.mesh_sr
        eri_lr = get_eri_lr_aftdf(lrdf, kpts=kpts, compact=compact,
                                  bvk_kmesh=bvk_kmesh)
        aosym = 's2' if eri_lr.shape[0] == nao_fat2_s2 else 's1'
        # zero out dd-related blocks
        cc_idx, cd_idx, dd_idx = get_aop_idx(cell_fat, aosym)
        eri_lr[dd_idx] = 0.
        eri_lr[:,dd_idx] = 0.

    t1 = log.timer_debug1('compact lr eri', *t1)

    # then compute full eri
    lrdf = df.AFTDF(cell_fat)
    lrdf.mesh = mydf.mesh_lr
    eri_lr2 = get_eri_lr_aftdf(lrdf, kpts=kpts, compact=compact,
                               bvk_kmesh=bvk_kmesh)
    # zero out non-dd blocks
    eri_lr2[np.ix_(cc_idx,cc_idx)] = 0
    eri_lr2[np.ix_(cc_idx,cd_idx)] = 0
    eri_lr2[np.ix_(cd_idx,cc_idx)] = 0
    eri_lr2[np.ix_(cd_idx,cd_idx)] = 0

    # add to the compact long-range part
    eri_lr2 += eri_lr

    t1 = log.timer_debug1('diffuse lr eri', *t1)

    # cell_fat to cell
    nao2 = nao*nao
    nao2_s2 = nao*(nao+1)//2
    eri_lr = np.zeros((nao2,nao2), dtype=eri_lr2.dtype)
    if aosym == 's2':
        eri_lr2 = ao2mo.restore(1, eri_lr2, nao_fat).reshape(nao_fat2,nao_fat2)

    for iap_fat, iap in fat_orig_loop(cell_fat, cell, 's1'):
        for jap_fat, jap in fat_orig_loop(cell_fat, cell, 's1'):
            eri_lr[np.ix_(iap,jap)] += eri_lr2[np.ix_(iap_fat,jap_fat)]

    if aosym == 's2':
        eri_lr = ao2mo.restore(4, eri_lr, nao)

    t1 = log.timer_debug1('fat2orig lr eri', *t1)

    return eri_lr


def get_eri_lr_aftdf(mydf, kpts=None,
                     compact=getattr(__config__,
                                     'pbc_df_ao2mo_get_eri_compact', True),
                     bvk_kmesh=None):
    cell = mydf.cell
    nao = cell.nao_nr()
    kptijkl = _format_kpts(kpts)
    if not _iskconserv(cell, kptijkl):
        lib.logger.warn(cell, 'aft_ao2mo: momentum conservation not found in '
                        'the given k-points %s', kptijkl)
        return np.zeros((nao,nao,nao,nao))

    kpti, kptj, kptk, kptl = kptijkl
    q = kptj - kpti
    mesh = mydf.mesh
    coulG = mydf.weighted_coulG(q, False, mesh)
    nao_pair = nao * (nao+1) // 2
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]) * .8)

####################
# gamma point, the integral is real and with s4 symmetry
    if gamma_point(kptijkl):
        eriR = np.zeros((nao_pair,nao_pair))
        for pqkR, pqkI, p0, p1 \
                in mydf.pw_loop(mesh, kptijkl[:2], q, max_memory=max_memory,
                                aosym='s2', bvk_kmesh=bvk_kmesh):
            lib.ddot(pqkR*coulG[p0:p1], pqkR.T, 1, eriR, 1)
            lib.ddot(pqkI*coulG[p0:p1], pqkI.T, 1, eriR, 1)
            pqkR = pqkI = None
        if not compact:
            eriR = ao2mo.restore(1, eriR, nao).reshape(nao**2,-1)
        return eriR

####################
# (kpt) i == j == k == l != 0
# (kpt) i == l && j == k && i != j && j != k  =>
#
# complex integrals, N^4 elements
    elif is_zero(kpti-kptl) and is_zero(kptj-kptk):
        eriR = np.zeros((nao**2,nao**2))
        eriI = np.zeros((nao**2,nao**2))
        for pqkR, pqkI, p0, p1 \
                in mydf.pw_loop(mesh, kptijkl[:2], q, max_memory=max_memory,
                    bvk_kmesh=bvk_kmesh):
# rho_pq(G+k_pq) * conj(rho_rs(G-k_rs))
            zdotNC(pqkR*coulG[p0:p1], pqkI*coulG[p0:p1], pqkR.T, pqkI.T,
                   1, eriR, eriI, 1)
            pqkR = pqkI = None
        pqkR = pqkI = coulG = None
# transpose(0,1,3,2) because
# j == k && i == l  =>
# (L|ij).transpose(0,2,1).conj() = (L^*|ji) = (L^*|kl)  =>  (M|kl)
# rho_rs(-G+k_rs) = conj(transpose(rho_sr(G+k_sr), (0,2,1)))
        eri = lib.transpose((eriR+eriI*1j).reshape(-1,nao,nao), axes=(0,2,1))
        return eri.reshape(nao**2,-1)

####################
# aosym = s1, complex integrals
#
# If kpti == kptj, (kptl-kptk)*a has to be multiples of 2pi because of the wave
# vector symmetry.  k is a fraction of reciprocal basis, 0 < k/b < 1, by definition.
# So  kptl/b - kptk/b  must be -1 < k/b < 1.  =>  kptl == kptk
#
    else:
        eriR = np.zeros((nao**2,nao**2))
        eriI = np.zeros((nao**2,nao**2))
#
#       (pq|rs) = \sum_G 4\pi rho_pq rho_rs / |G+k_{pq}|^2
#       rho_pq = 1/N \sum_{Tp,Tq} \int exp(-i(G+k_{pq})*r) p(r-Tp) q(r-Tq) dr
#              = \sum_{Tq} exp(i k_q*Tq) \int exp(-i(G+k_{pq})*r) p(r) q(r-Tq) dr
# Note the k-point wrap-around for rho_rs, which leads to G+k_{pq} in FT
#       rho_rs = 1/N \sum_{Tr,Ts} \int exp( i(G+k_{pq})*r) r(r-Tr) s(r-Ts) dr
#              = \sum_{Ts} exp(i k_s*Ts) \int exp( i(G+k_{pq})*r) r(r) s(r-Ts) dr
# rho_pq can be directly evaluated by AFT (function pw_loop)
#       rho_pq = pw_loop(k_q, G+k_{pq})
# Assuming r(r) and s(r) are real functions, rho_rs is evaluated
#       rho_rs = 1/N \sum_{Tr,Ts} \int exp( i(G+k_{pq})*r) r(r-Tr) s(r-Ts) dr
#              = conj(\sum_{Ts} exp(-i k_s*Ts) \int exp(-i(G+k_{pq})*r) r(r) s(r-Ts) dr)
#              = conj( pw_loop(-k_s, G+k_{pq}) )
#
# TODO: For complex AO function r(r) and s(r), pw_loop function needs to be
# extended to include Gv vector in the arguments
        for (pqkR, pqkI, p0, p1), (rskR, rskI, q0, q1) in \
                lib.izip(mydf.pw_loop(mesh, kptijkl[:2], q, max_memory=max_memory*.5, bvk_kmesh=bvk_kmesh),
                         mydf.pw_loop(mesh,-kptijkl[2:], q, max_memory=max_memory*.5, bvk_kmesh=bvk_kmesh)):
            pqkR *= coulG[p0:p1]
            pqkI *= coulG[p0:p1]
            zdotNC(pqkR, pqkI, rskR.T, rskI.T, 1, eriR, eriI, 1)
            pqkR = pqkI = rskR = rskI = None
        return (eriR+eriI*1j)
