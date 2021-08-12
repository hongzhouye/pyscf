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

import copy
import numpy as np
import scipy.linalg

from pyscf.pbc.df.df import LINEAR_DEP_THR
from pyscf.pbc.df.rsdf_helper import intor_j2c
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member
from pyscf import lib
from pyscf.lib import logger

def density_fit(mf, auxbasis=None, mesh=None, with_df=None):
    '''Generte density-fitting SCF object

    Args:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.  If auxbasis is
            None, auxiliary basis based on AO basis (if possible) or
            even-tempered Gaussian basis will be used.
        mesh : tuple
            number of grids in each direction
        with_df : DF object
    '''
    from pyscf.pbc.df import rsdf
    if with_df is None:
        if getattr(mf, 'kpts', None) is not None:
            kpts = mf.kpts
        else:
            kpts = np.reshape(mf.kpt, (1,3))

        with_df = rsdf.RSDF(mf.cell, kpts)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis
        if mesh is not None:
            with_df.mesh = mesh

    mf = copy.copy(mf)
    mf.with_df = with_df
    mf._eri = None
    return mf

def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None, bvk_kmesh=None):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

# j2c: For j-build, only j2c at gamma point is needed.
    omega = abs(mydf.omega)
    j2c = intor_j2c(mydf.auxcell, omega)
    j2c_inv = invert_j2c(j2c, mydf.linear_dep_threshold)

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    if mydf.auxcell is None:
        # If mydf._cderi is the file that generated from another calculation,
        # guess naux based on the contents of the integral file.
        naux = mydf.get_naoaux()
    else:
        naux = mydf.auxcell.nao_nr()
    nao_pair = nao * (nao+1) // 2

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    j_real = gamma_point(kpts_band) and not np.iscomplexobj(dms)

    kptiis = np.repeat(kpts, 2, axis=0).reshape(-1,2,3)

    dmsR = dms.real.transpose(0,1,3,2).reshape(nset,nkpts,nao**2)
    dmsI = dms.imag.transpose(0,1,3,2).reshape(nset,nkpts,nao**2)
    rhoR = np.zeros((nset,naux))
    rhoI = np.zeros((nset,naux))
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]))

    p1_kpts = np.zeros(nkpts, dtype=int)
    for istep, k, LpqR, LpqI, sign in mydf.sr_loop(kptij_lst=kptiis,
                max_memory=max_memory, compact=False, bvk_kmesh=bvk_kmesh):
        p1 = p1_kpts[k]
        p0, p1 = p1, p1+LpqR.shape[0]
        p1_kpts[k] = p1
        tmp = sign * np.einsum('Lp,xp->xL', LpqR, dmsR[:,k])
        rhoR += np.dot(tmp, j2c_inv[p0:p1])
        tmp = sign * np.einsum('Lp,xp->xL', LpqR, dmsI[:,k])
        rhoI += np.dot(tmp, j2c_inv[p0:p1])
        if not LpqI is None:
            tmp = sign * np.einsum('Lp,xp->xL', LpqI, dmsI[:,k])
            rhoR -= np.dot(tmp, j2c_inv[p0:p1])
            tmp = sign * np.einsum('Lp,xp->xL', LpqI, dmsR[:,k])
            rhoI += np.dot(tmp, j2c_inv[p0:p1])
        LpqR = LpqI = tmp = None
    t1 = log.timer_debug1('get_j pass 1', *t1)

    weight = 1./nkpts
    rhoR *= weight
    rhoI *= weight
    vjR = np.zeros((nset,nband,nao_pair))
    vjI = np.zeros((nset,nband,nao_pair))

    kptiis_band = np.repeat(kpts_band, 2, axis=0).reshape(-1,2,3)

    p1_kpts = np.zeros(nband, dtype=int)
    for istep, k, LpqR, LpqI, sign in mydf.sr_loop(kptij_lst=kptiis_band,
                max_memory=max_memory, compact=True, bvk_kmesh=bvk_kmesh):
        p1 = p1_kpts[k]
        p0, p1 = p1, p1+LpqR.shape[0]
        p1_kpts[k] = p1
        vjR[:,k] += np.dot(rhoR[:,p0:p1], LpqR)
        if not j_real:
            vjI[:,k] += np.dot(rhoI[:,p0:p1], LpqR)
            if LpqI is not None:
                vjR[:,k] -= np.dot(rhoI[:,p0:p1], LpqI)
                vjI[:,k] += np.dot(rhoR[:,p0:p1], LpqI)
        LpqR = LpqI = None
    t1 = log.timer_debug1('get_j pass 2', *t1)

    if j_real:
        vj_kpts = vjR
    else:
        vj_kpts = vjR + vjI*1j
    vj_kpts = lib.unpack_tril(vj_kpts.reshape(-1,nao_pair))
    vj_kpts = vj_kpts.reshape(nset,nband,nao,nao)

    return _format_jks(vj_kpts, dm_kpts, input_band, kpts)

def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
               exxdiv=None, bvk_kmesh=None):
    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)

    if exxdiv is not None and exxdiv != 'ewald':
        log.warn('GDF does not support exxdiv %s. '
                 'exxdiv needs to be "ewald" or None', exxdiv)
        raise RuntimeError('GDF does not support exxdiv %s' % exxdiv)

    t1 = (logger.process_clock(), logger.perf_counter())

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    vkR = np.zeros((nset,nband,nao,nao))
    vkI = np.zeros((nset,nband,nao,nao))
    dmsR = np.asarray(dms.real, order='C')
    dmsI = np.asarray(dms.imag, order='C')

    # K_pq = ( p{k1} i{k2} | i{k2} q{k1} )
    bufR = np.empty((mydf.blockdim*nao**2))
    bufI = np.empty((mydf.blockdim*nao**2))
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    def make_kpt(ki, kj, swap_2e):
        kpti = kpts[ki]
        kptj = kpts_band[kj]

        for LpqR, LpqI, sign in mydf.sr_loop((kpti,kptj), max_memory, False):
            nrow = LpqR.shape[0]
            pLqR = np.ndarray((nao,nrow,nao), buffer=bufR)
            pLqI = np.ndarray((nao,nrow,nao), buffer=bufI)
            tmpR = np.ndarray((nao,nrow*nao), buffer=LpqR)
            tmpI = np.ndarray((nao,nrow*nao), buffer=LpqI)
            pLqR[:] = LpqR.reshape(-1,nao,nao).transpose(1,0,2)
            pLqI[:] = LpqI.reshape(-1,nao,nao).transpose(1,0,2)

            for i in range(nset):
                zdotNN(dmsR[i,ki], dmsI[i,ki], pLqR.reshape(nao,-1),
                       pLqI.reshape(nao,-1), 1, tmpR, tmpI)
                zdotCN(pLqR.reshape(-1,nao).T, pLqI.reshape(-1,nao).T,
                       tmpR.reshape(-1,nao), tmpI.reshape(-1,nao),
                       sign, vkR[i,kj], vkI[i,kj], 1)

            if swap_2e:
                tmpR = tmpR.reshape(nao*nrow,nao)
                tmpI = tmpI.reshape(nao*nrow,nao)
                for i in range(nset):
                    zdotNN(pLqR.reshape(-1,nao), pLqI.reshape(-1,nao),
                           dmsR[i,kj], dmsI[i,kj], 1, tmpR, tmpI)
                    zdotNC(tmpR.reshape(nao,-1), tmpI.reshape(nao,-1),
                           pLqR.reshape(nao,-1).T, pLqI.reshape(nao,-1).T,
                           sign, vkR[i,ki], vkI[i,ki], 1)

    if kpts_band is kpts:  # normal k-points HF/DFT
        for ki in range(nkpts):
            for kj in range(ki):
                make_kpt(ki, kj, True)
            make_kpt(ki, ki, False)
            t1 = log.timer_debug1('get_k_kpts: make_kpt ki>=kj (%d,*)'%ki, *t1)
    else:
        for ki in range(nkpts):
            for kj in range(nband):
                make_kpt(ki, kj, False)
            t1 = log.timer_debug1('get_k_kpts: make_kpt (%d,*)'%ki, *t1)

    if (gamma_point(kpts) and gamma_point(kpts_band) and
        not np.iscomplexobj(dm_kpts)):
        vk_kpts = vkR
    else:
        vk_kpts = vkR + vkI * 1j
    vk_kpts *= 1./nkpts

    if exxdiv == 'ewald':
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)


def invert_j2c(j2c, linear_dep_threshold=LINEAR_DEP_THR):
    w, v = scipy.linalg.eigh(j2c)
    mask_keep = w>linear_dep_threshold
    ndrop = np.count_nonzero(~mask_keep)
    if ndrop > 0:
        log.debug('DF metric linear dependency for kpt %s',
                  uniq_kptji_id)
        log.debug('cond = %.4g, drop %d bfns', w[-1]/w[0], ndrop)
        print("drop %d !!!" % ndrop)
    v1 = v[:,mask_keep]
    w1_inv = 1./w[mask_keep]
    j2c_inv = np.dot(v1*w1_inv, v1.T.conj())
    return j2c_inv
def cholesky_decomposed_metric(cell, j2c, eig_always=False,
                               linear_dep_threshold=LINEAR_DEP_THR):
    j2c_negative = None
    try:
        if eig_always:
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
        ndrop = np.count_nonzero(w<linear_dep_threshold)
        if ndrop > 0:
            log.debug('DF metric linear dependency for kpt %s',
                      uniq_kptji_id)
            log.debug('cond = %.4g, drop %d bfns', w[-1]/w[0], ndrop)
        v1 = v[:,w>linear_dep_threshold].conj().T
        v1 /= np.sqrt(w[w>linear_dep_threshold]).reshape(-1,1)
        j2c = v1
        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            idx = np.where(w < -linear_dep_threshold)[0]
            if len(idx) > 0:
                j2c_negative = (v[:,idx]/np.sqrt(-w[idx])).conj().T
        w = v = None
        j2ctag = 'eig'
    return j2c, j2c_negative, j2ctag
def j2c_contract(v, cholesky_j2c):
    j2c, j2c_negative, j2ctag = cholesky_j2c

    # low-dimension systems
    if j2c_negative is not None:
        raise NotImplementedError

    if j2ctag == 'CD':
        v = scipy.linalg.solve_triangular(j2c, v, lower=True, overwrite_b=True)
    else:
        v = lib.dot(j2c, v)
    return v
