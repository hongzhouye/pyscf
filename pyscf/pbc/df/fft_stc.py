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
#         Gengzhi Yang <genzyang17@gmail.com>
#


import numpy as np

from .fft import FFTDF
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.df import fft_jk
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc.lib.kpts_helper import is_zero


class FFTDF_STC(FFTDF):
    omega_dot_Rc = 4.

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None:  # J/K for RSH functionals
            raise NotImplementedError

        from pyscf.pbc.df.aft import _check_kpts
        kpts, is_single_kpt = _check_kpts(self, kpts)
        # if is_single_kpt:
        #     vj, vk = fft_jk.get_jk(self, dm, hermi, kpts[0], kpts_band,
        #                            with_j, with_k, exxdiv)
        # else:
        if True:
            vj = vk = None
            if with_k:
                vk = get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
            if with_j:
                vj = fft_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk


def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
               exxdiv=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray
            Density matrix at each k-point
        kpts : (nkpts, 3) ndarray

    Kwargs:
        hermi : int
            Whether K matrix is hermitian

            | 0 : not hermitian and not symmetric
            | 1 : hermitian

        kpts_band : ``(3,)`` ndarray or ``(*,3)`` ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        vk : (nkpts, nao, nao) ndarray
        or list of vj and vk if the input dm_kpts is a list of DMs
    '''
    log = logger.new_logger(mydf)
    cell = mydf.cell
    mesh = mydf.mesh
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension != 1
    coords = cell.gen_uniform_grids(mesh)
    ngrids = coords.shape[0]

    if getattr(dm_kpts, 'mo_coeff', None) is not None:
        mo_coeff = dm_kpts.mo_coeff
        mo_occ   = dm_kpts.mo_occ
    else:
        mo_coeff = None

    kpts = np.asarray(kpts)
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    weight = 1./nkpts * (cell.vol/ngrids)

    Rc = (3*nkpts*cell.vol/(4*np.pi))**(1./3)
    omega_stc = mydf.omega_dot_Rc / Rc
    log.warn('omega_stc = %.10f', omega_stc)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    if is_zero(kpts_band) and is_zero(kpts):
        vk_kpts = np.zeros((nset,nband,nao,nao), dtype=dms.dtype)
    else:
        vk_kpts = np.zeros((nset,nband,nao,nao), dtype=np.complex128)

    coords = mydf.grids.coords
    ao2_kpts = [np.asarray(ao.T, order='C')
                for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts)]
    if input_band is None:
        ao1_kpts = ao2_kpts
    else:
        ao1_kpts = [np.asarray(ao.T, order='C')
                    for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts_band)]
    if mo_coeff is not None and nset == 1:
        mo_coeff = [mo_coeff[k][:,occ>0] * np.sqrt(occ[occ>0])
                    for k, occ in enumerate(mo_occ)]
        ao2_kpts = [np.dot(mo_coeff[k].T, ao) for k, ao in enumerate(ao2_kpts)]

    mem_now = lib.current_memory()[0]
    max_memory = mydf.max_memory - mem_now
    blksize = int(min(nao, max(1, (max_memory-mem_now)*1e6/16/4/ngrids/nao)))
    log.debug1('fft_jk: get_k_kpts max_memory %s  blksize %d', max_memory, blksize)
    #ao1_dtype = np.result_type(*ao1_kpts)
    #ao2_dtype = np.result_type(*ao2_kpts)
    vR_dm = np.empty((nset,nao,ngrids), dtype=vk_kpts.dtype)

    t1 = log.init_timer()
    for k2, ao2T in enumerate(ao2_kpts):
        if ao2T.size == 0:
            continue

        kpt2 = kpts[k2]
        naoj = ao2T.shape[0]
        if mo_coeff is None or nset > 1:
            ao_dms = [lib.dot(dms[i,k2], ao2T.conj()) for i in range(nset)]
        else:
            ao_dms = [ao2T.conj()]

        for k1, ao1T in enumerate(ao1_kpts):
            kpt1 = kpts_band[k1]

            # In PySCF v1.5 - v2.12, the G=0 term is evaluated analytically
            # using _ewald_exxdiv_for_G0. The G=0 component obtained here may
            # differ from _ewald_exxdiv_for_G0 due to discretization errors in
            # the FFT-based density (especially when the mesh is not
            # sufficiently dense), which can lead to small discrepancies.
            coulG = get_coulG(cell, kpt2-kpt1, exxdiv, mydf, mesh, omega=omega_stc)
            if is_zero(kpt1-kpt2):
                expmikr = np.array(1.)
            else:
                expmikr = np.exp(-1j * np.dot(coords, kpt2-kpt1))

            for p0, p1 in lib.prange(0, nao, blksize):
                rho1 = np.einsum('ig,jg->ijg', ao1T[p0:p1].conj()*expmikr, ao2T)
                vG = tools.fft(rho1.reshape(-1,ngrids), mesh)
                rho1 = None
                vG *= coulG
                vR = tools.ifft(vG, mesh).reshape(p1-p0,naoj,ngrids)
                vG = None
                if vR_dm.dtype == np.double:
                    vR = vR.real
                for i in range(nset):
                    np.einsum('ijg,jg->ig', vR, ao_dms[i], out=vR_dm[i,p0:p1])
                vR = None
            vR_dm *= expmikr.conj()

            for i in range(nset):
                vk_kpts[i,k1] += weight * lib.dot(vR_dm[i], ao1T.T)
        t1 = log.timer_debug1('get_k_kpts: make_kpt (%d,*)'%k2, *t1)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)


def get_coulG(cell, k=np.zeros(3), exx=False, mf=None, mesh=None, Gv=None,
              wrap_around=True, omega=None, **kwargs):
    assert( isinstance(exx, str) )
    assert( exx.lower() in ['vcut_sph', 'vcut_ws'] )
    assert( omega is not None )

    # calculate vcut coulG without omega
    coulG = tools.get_coulG(cell, k, exx, mf, mesh, Gv, wrap_around, None, **kwargs)

    # smooth modification with omega
    if mesh is None:
        mesh = cell.mesh
    if 'gs' in kwargs:
        warnings.warn('cell.gs is deprecated.  It is replaced by cell.mesh,'
                      'the number of PWs (=2*gs+1) along each direction.')
        mesh = [2*n+1 for n in kwargs['gs']]
    if Gv is None:
        Gv = cell.get_Gv(mesh)

    if abs(k).sum() > 1e-9:
        if wrap_around:
            # Here we 'wrap around' the high frequency k+G vectors into their lower
            # frequency counterparts.  Important if you want the gamma point and k-point
            # answers to agree
            from pyscf.pbc.tools.pbc import _Gv_wrap_around
            kG = _Gv_wrap_around(cell, Gv, k, mesh)
        else:
            kG = k + Gv
    else:
        kG = Gv

    absG2 = np.einsum('gi,gi->g', kG, kG)

    if hasattr(mf, 'kpts'):
        kpts = mf.kpts
    else:
        kpts = k.reshape(1,3)
    Nk = len(kpts)

    f = np.exp(-absG2*0.25/omega**2.)

    v0 = coulG[absG2==0]
    coulG *= f
    with np.errstate(divide='ignore',invalid='ignore'):
        coulG += 4*np.pi/absG2 * (1. - f)
    coulG[absG2==0] = v0 + np.pi/omega**2.

    return coulG
