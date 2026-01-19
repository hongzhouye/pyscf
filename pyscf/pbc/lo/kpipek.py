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
#         Gengzhi Yang <genzyang17@gmail.com>
#

'''
K-point Pipek-Mezey localization

ref. [To be updated]
'''

import numpy
from functools import reduce

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.soscf import kciah
from pyscf.lo import orth, cholesky_mos
from pyscf.lo import boys
from pyscf.lo import iao
from pyscf.lo.stability import stability_newton
from pyscf.pbc.lo.stability import stability_jacobi
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.tools import k2gamma
from pyscf.tools import mo_mapping
from pyscf import __config__


def atomic_pops_contract(cell, mo_coeff, kpts, exponent, method='meta_lowdin', proj_data=None):
    '''
    Kwargs:
        method : string
            The atomic population projection scheme. It can be mulliken,
            lowdin, meta_lowdin, iao, or becke

    Returns:
        A 3-index tensor [A,i,j] indicates the population of any orbital-pair
        density |i><j| for each species (atom in this case).  This tensor is
        used to construct the population and gradients etc.

        You can customize the PM localization wrt other population metric,
        such as the charge of a site, the charge of a fragment (a group of
        atoms) by overwriting this tensor.  See also the example
        pyscf/examples/loc_orb/40-hubbard_model_PM_localization.py for the PM
        localization of site-based population for hubbard model.
    '''
    method = method.lower().replace('_', '-')
    mo_coeff = numpy.asarray(mo_coeff)
    nkpts,nao,nmo = mo_coeff.shape
    kmesh = get_kmesh(cell, kpts)
    scell, phase = k2gamma.get_phase(cell, kpts, kmesh=kmesh)

    def contract_orth(mo_coeff, proj_coeff, s, offset_nr_by_atom):
        kcsc = lib.einsum('kmi,kmn,knSx->kiSx', mo_coeff.conj(), s, proj_coeff)
        scsc = kcsc.sum(axis=0).reshape(nmo,-1)
        kcsc = kcsc.reshape(nkpts*nmo,-1)

        proj0k = numpy.zeros((scell.natm,nmo,nkpts,nmo), dtype=numpy.complex128)
        for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
            proj1 = numpy.dot(scsc[:,p0:p1], kcsc[:,p0:p1].conj().T)
            proj0k[i] = proj1.reshape(nmo,nkpts,nmo)

        popkk = numpy.zeros((scell.natm,nkpts,nmo), dtype=numpy.float64)
        QP = numpy.zeros((nkpts,nkpts,nmo,nmo,nmo), dtype=numpy.complex128)
        buf = numpy.empty((nkpts,nkpts,nmo,nmo), dtype=QP.dtype)
        for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
            proj1 = numpy.dot(kcsc[:,p0:p1], kcsc[:,p0:p1].conj().T)
            buf[:] = proj1.reshape(nkpts,nmo,nkpts,nmo).transpose(0,2,1,3)
            popkk[i] = lib.einsum('kkii->ki', buf.real)
            popexp1 = lib.einsum('ktii->i', buf.real)**(exponent-1)
            lib.outer(buf.reshape(-1), popexp1, out=QP.reshape(-1, nmo))
        buf = None

        return QP, proj0k, popkk

    def contract_biorth(mo_coeff, proj_coeff, projtild_coeff, s, offset_nr_by_atom):
        kcsc = lib.einsum('kmi,kmn,knSx->kiSx', mo_coeff.conj(), s, proj_coeff)
        kcsctild = lib.einsum('kmi,kmn,knSx->kiSx', mo_coeff.conj(), s, projtild_coeff)
        scsc = kcsc.sum(axis=0).reshape(nmo,-1)
        scsctild = kcsctild.sum(axis=0).reshape(nmo,-1)
        kcsc = kcsc.reshape(nkpts*nmo,-1)
        kcsctild = kcsctild.reshape(nkpts*nmo,-1)

        proj0k = numpy.empty((scell.natm,nmo,nkpts,nmo), dtype=mo_coeff.dtype)
        for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
            proj1  = numpy.dot(scsc[:,p0:p1], kcsctild[:,p0:p1].conj().T)
            proj1 += numpy.dot(scsctild[:,p0:p1], kcsc[:,p0:p1].conj().T)
            proj0k[i] = proj1.reshape(nmo,nkpts,nmo) * 0.5

        popkk = numpy.zeros((scell.natm,nkpts,nmo), dtype=numpy.float64)
        QP = numpy.zeros((nkpts,nkpts,nmo,nmo,nmo), dtype=numpy.complex128)
        buf = numpy.empty((nkpts,nkpts,nmo,nmo), dtype=QP.dtype)
        for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
            proj1  = numpy.dot(kcsc[:,p0:p1], kcsctild[:,p0:p1].conj().T)
            proj1 += numpy.dot(kcsctild[:,p0:p1], kcsc[:,p0:p1].conj().T)
            buf[:] = proj1.reshape(nkpts,nmo,nkpts,nmo).transpose(0,2,1,3) * 0.5
            popkk[i] = lib.einsum('kkii->ki', buf.real)
            popexp1 = lib.einsum('ktii->i', buf.real)**(exponent-1)
            lib.outer(buf.reshape(-1), popexp1, out=QP.reshape(-1, nmo))
        buf = None

        return QP, proj0k, popkk

    if method == 'mulliken':
        raise NotImplementedError
        s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
        proj_coeff = lib.einsum('Rk,kmi->Rmki', phase, mo_coeff) / nkpts**0.5
        proj_coeff = proj_coeff.reshape(-1,nkpts*nmo)
        s_scell = lib.einsum('Rk,kmn,Sk->RmSn', phase, s, phase.conj()).reshape(nkpts*nao,-1)
        if abs(s_scell.imag).max() < 1e-10:
            s_scell = s_scell.real
        for i, (b0, b1, p0, p1) in enumerate(scell.offset_nr_by_atom()):
            proj1 = reduce(numpy.dot, (proj_coeff[p0:p1].conj().T, s_scell[p0:p1], proj_coeff))
            proj1 += proj1.conj().T
            proj[i] = proj1.reshape(nkpts,nmo,nkpts,nmo).transpose(0,2,1,3) * 0.5

    elif method in ('lowdin', 'meta-lowdin'):
        if proj_data is None:
            s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
            proj_coeff = numpy.asarray([orth.orth_ao(cell, method, 'ANO', s=s[k],
                                        adjust_phase=False) for k in range(nkpts)])
            proj_coeff = lib.einsum('kmx,Sk->kmSx', proj_coeff, phase.conj()) / nkpts**0.5
            offset_nr_by_atom = scell.offset_nr_by_atom()
        else:
            proj_coeff, s, offset_nr_by_atom = proj_data

        QP, proj0k, popkk = contract_orth(mo_coeff, proj_coeff, s, offset_nr_by_atom)

    elif method in ('iao', 'ibo', 'iao-biorth'):
        if proj_data is None:
            s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
            iao_coeff = iao.iao(cell, mo_coeff, kpts=kpts)
            iao_scell = iao.reference_mol(scell)
            offset_nr_by_atom = iao_scell.offset_nr_by_atom()

        if method == 'iao-biorth':
            if proj_data is None:
                proj_coeff = lib.einsum('kmx,Sk->kmSx', iao_coeff, phase.conj()) / nkpts**0.5
                ovlp = lib.einsum('kmx,kmn,kny->kxy', iao_coeff.conj(), s, iao_coeff)
                iaotild_coeff = numpy.asarray([numpy.linalg.solve(ovlp[k],
                                               iao_coeff[k].conj().T).conj().T
                                               for k in range(nkpts)], order='C')
                projtild_coeff = lib.einsum('kmx,Sk->kmSx', iaotild_coeff,
                                            phase.conj()) / nkpts**0.5
            else:
                proj_coeff, projtild_coeff, s, offset_nr_by_atom = proj_data

            QP, proj0k, popkk = contract_biorth(mo_coeff, proj_coeff, projtild_coeff, s,
                                                offset_nr_by_atom)
        else:
            if proj_data is None:
                proj_coeff = numpy.asarray([orth.vec_lowdin(iao_coeff[k], s[k])
                                            for k in range(nkpts)])
                proj_coeff = lib.einsum('kmx,Sk->kmSx', proj_coeff, phase.conj()) / nkpts**0.5
            else:
                proj_coeff, s, offset_nr_by_atom = proj_data

            QP, proj0k, popkk = contract_orth(mo_coeff, proj_coeff, s, offset_nr_by_atom)

    else:
        raise KeyError('method = %s' % method)

    return QP, proj0k, popkk


def atomic_pops(cell, mo_coeff, kpts, mode='kk', method='meta_lowdin', proj_data=None):
    '''
    Kwargs:
        method : string
            The atomic population projection scheme. It can be mulliken,
            lowdin, meta_lowdin, iao, or becke

    Returns:
        A 3-index tensor [A,i,j] indicates the population of any orbital-pair
        density |i><j| for each species (atom in this case).  This tensor is
        used to construct the population and gradients etc.

        You can customize the PM localization wrt other population metric,
        such as the charge of a site, the charge of a fragment (a group of
        atoms) by overwriting this tensor.  See also the example
        pyscf/examples/loc_orb/40-hubbard_model_PM_localization.py for the PM
        localization of site-based population for hubbard model.
    '''
    method = method.lower().replace('_', '-')
    mo_coeff = numpy.asarray(mo_coeff)
    nkpts,nao,nmo = mo_coeff.shape
    kmesh = get_kmesh(cell, kpts)
    scell, phase = k2gamma.get_phase(cell, kpts, kmesh=kmesh)

    def proj_orth(mo_coeff, proj_coeff, s, offset_nr_by_atom):
        if mode == 'kk':
            proj = numpy.empty((scell.natm,nkpts,nmo,nkpts,nmo), dtype=mo_coeff.dtype)
            kcsc = lib.einsum('kmi,kmn,knSx->kiSx', mo_coeff.conj(), s, proj_coeff)
            kcsc = kcsc.reshape(nkpts*nmo,-1)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                proj1 = numpy.dot(kcsc[:,p0:p1], kcsc[:,p0:p1].conj().T)
                proj[i] = proj1.reshape(nkpts,nmo,nkpts,nmo)

        elif mode in ['0k','k0']:
            kcsc = lib.einsum('kmi,kmn,knSx->kiSx', mo_coeff.conj(), s, proj_coeff)
            scsc = kcsc.sum(axis=0).reshape(nmo,-1)
            kcsc = kcsc.reshape(nkpts*nmo,-1)

            if mode == '0k':
                proj = numpy.empty((scell.natm,nmo,nkpts,nmo), dtype=mo_coeff.dtype)
                for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                    proj1 = numpy.dot(scsc[:,p0:p1], kcsc[:,p0:p1].conj().T)
                    proj[i] = proj1.reshape(nmo,nkpts,nmo)
            else:
                proj = numpy.empty((scell.natm,nkpts,nmo,nmo), dtype=mo_coeff.dtype)
                for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                    proj1 = numpy.dot(kcsc[:,p0:p1], scsc[:,p0:p1].conj().T)
                    proj[i] = proj1.reshape(nkpts,nmo,nmo)

        elif mode == '00':
            proj = numpy.empty((scell.natm,nmo,nmo), dtype=mo_coeff.dtype)
            scsc = lib.einsum('kmi,kmn,knSx->iSx', mo_coeff.conj(), s, proj_coeff)
            scsc = scsc.reshape(nmo,-1)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                proj[i] = numpy.dot(scsc[:,p0:p1], scsc[:,p0:p1].conj().T)

        else:
            raise ValueError('Unknown mode %s' % str(mode))

        return proj

    def proj_biorth(mo_coeff, proj_coeff, projtild_coeff, s, offset_nr_by_atom):
        if mode == 'kk':
            proj = numpy.empty((scell.natm,nkpts,nmo,nkpts,nmo), dtype=mo_coeff.dtype)
            kcsc = lib.einsum('kmi,kmn,knSx->kiSx', mo_coeff.conj(), s, proj_coeff)
            kcsctild = lib.einsum('kmi,kmn,knSx->kiSx', mo_coeff.conj(), s, projtild_coeff)
            kcsc = kcsc.reshape(nkpts*nmo,-1)
            kcsctild = kcsctild.reshape(nkpts*nmo,-1)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                proj1  = numpy.dot(kcsc[:,p0:p1], kcsctild[:,p0:p1].conj().T)
                proj1 += numpy.dot(kcsctild[:,p0:p1], kcsc[:,p0:p1].conj().T)
                proj[i] = proj1.reshape(nkpts,nmo,nkpts,nmo) * 0.5

        elif mode in ['0k','k0']:
            kcsc = lib.einsum('kmi,kmn,knSx->kiSx', mo_coeff.conj(), s, proj_coeff)
            kcsctild = lib.einsum('kmi,kmn,knSx->kiSx', mo_coeff.conj(), s, projtild_coeff)
            scsc = kcsc.sum(axis=0).reshape(nmo,-1)
            scsctild = kcsctild.sum(axis=0).reshape(nmo,-1)
            kcsc = kcsc.reshape(nkpts*nmo,-1)
            kcsctild = kcsctild.reshape(nkpts*nmo,-1)

            if mode == '0k':
                proj = numpy.empty((scell.natm,nmo,nkpts,nmo), dtype=mo_coeff.dtype)
                for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                    proj1  = numpy.dot(scsc[:,p0:p1], kcsctild[:,p0:p1].conj().T)
                    proj1 += numpy.dot(scsctild[:,p0:p1], kcsc[:,p0:p1].conj().T)
                    proj[i] = proj1.reshape(nmo,nkpts,nmo) * 0.5
            else:
                proj = numpy.empty((scell.natm,nkpts,nmo,nmo), dtype=mo_coeff.dtype)
                for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                    proj1  = numpy.dot(kcsc[:,p0:p1], scsctild[:,p0:p1].conj().T)
                    proj1 += numpy.dot(kcsctild[:,p0:p1], scsc[:,p0:p1].conj().T)
                    proj[i] = proj1.reshape(nmo,nkpts,nmo) * 0.5

        elif mode == '00':
            proj = numpy.empty((scell.natm,nmo,nmo), dtype=mo_coeff.dtype)
            scsc = lib.einsum('kmi,kmn,knSx->iSx', mo_coeff.conj(), s, proj_coeff)
            scsctild = lib.einsum('kmi,kmn,knSx->iSx', mo_coeff.conj(), s, projtild_coeff)
            scsc = scsc.reshape(nmo,-1)
            scsctild = scsctild.reshape(nmo,-1)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                proj1  = numpy.dot(scsc[:,p0:p1], scsctild[:,p0:p1].conj().T)
                proj1 += numpy.dot(scsctild[:,p0:p1], scsc[:,p0:p1].conj().T)
                proj[i] = proj1 * 0.5

        else:
            raise ValueError('Unknown mode %s' % str(mode))

        return proj

    if method == 'mulliken':
        s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
        proj_coeff = lib.einsum('Rk,kmi->Rmki', phase, mo_coeff) / nkpts**0.5
        proj_coeff = proj_coeff.reshape(-1,nkpts*nmo)
        s_scell = lib.einsum('Rk,kmn,Sk->RmSn', phase, s, phase.conj()).reshape(nkpts*nao,-1)
        if abs(s_scell.imag).max() < 1e-10:
            s_scell = s_scell.real
        for i, (b0, b1, p0, p1) in enumerate(scell.offset_nr_by_atom()):
            proj1 = reduce(numpy.dot, (proj_coeff[p0:p1].conj().T, s_scell[p0:p1], proj_coeff))
            proj1 += proj1.conj().T
            proj[i] = proj1.reshape(nkpts,nmo,nkpts,nmo).transpose(0,2,1,3) * 0.5

    elif method in ('lowdin', 'meta-lowdin'):
        if proj_data is None:
            s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
            proj_coeff = numpy.asarray([orth.orth_ao(cell, method, 'ANO', s=s[k],
                                        adjust_phase=False) for k in range(nkpts)])
            proj_coeff = lib.einsum('kmx,Sk->kmSx', proj_coeff, phase.conj()) / nkpts**0.5
            offset_nr_by_atom = scell.offset_nr_by_atom()
        else:
            proj_coeff, s, offset_nr_by_atom = proj_data

        proj = proj_orth(mo_coeff, proj_coeff, s, offset_nr_by_atom)

    elif method in ('iao', 'ibo', 'iao-biorth'):
        if proj_data is None:
            s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
            iao_coeff = iao.iao(cell, mo_coeff, kpts=kpts)
            iao_scell = iao.reference_mol(scell)
            offset_nr_by_atom = iao_scell.offset_nr_by_atom()

        if method == 'iao-biorth':
            if proj_data is None:
                proj_coeff = lib.einsum('kmx,Sk->kmSx', iao_coeff, phase.conj()) / nkpts**0.5
                ovlp = lib.einsum('kmx,kmn,kny->kxy', iao_coeff.conj(), s, iao_coeff)
                iaotild_coeff = numpy.asarray([numpy.linalg.solve(ovlp[k],
                                               iao_coeff[k].conj().T).conj().T
                                               for k in range(nkpts)], order='C')
                projtild_coeff = lib.einsum('kmx,Sk->kmSx', iaotild_coeff,
                                            phase.conj()) / nkpts**0.5
            else:
                proj_coeff, projtild_coeff, s, offset_nr_by_atom = proj_data

            proj = proj_biorth(mo_coeff, proj_coeff, projtild_coeff, s, offset_nr_by_atom)
        else:
            if proj_data is None:
                proj_coeff = numpy.asarray([orth.vec_lowdin(iao_coeff[k], s[k])
                                            for k in range(nkpts)])
                proj_coeff = lib.einsum('kmx,Sk->kmSx', proj_coeff, phase.conj()) / nkpts**0.5
            else:
                proj_coeff, s, offset_nr_by_atom = proj_data

            proj = proj_orth(mo_coeff, proj_coeff, s, offset_nr_by_atom)

    else:
        raise KeyError('method = %s' % method)

    return proj


class KptsOrbitalLocalizer(lib.StreamObject, kciah.SubspaceCIAHOptimizerMixin):

    conv_tol = getattr(__config__, 'pbc_lo_kpipek_KPipek_conv_tol', 1e-6)
    conv_tol_grad = getattr(__config__, 'pbc_lo_kpipek_KPipek_conv_tol_grad', None)
    max_cycle = getattr(__config__, 'pbc_lo_kpipek_KPipek_max_cycle', 100)
    max_iters = getattr(__config__, 'pbc_lo_kpipek_KPipek_max_iters', 20)
    max_stepsize = getattr(__config__, 'pbc_lo_kpipek_KPipek_max_stepsize', .05)
    ah_trust_region = getattr(__config__, 'pbc_lo_kpipek_KPipek_ah_trust_region', 3)
    ah_start_tol = getattr(__config__, 'pbc_lo_kpipek_KPipek_ah_start_tol', 1e9)
    ah_max_cycle = getattr(__config__, 'pbc_lo_kpipek_KPipek_ah_max_cycle', 40)
    init_guess = getattr(__config__, 'pbc_lo_kpipek_KPipek_init_guess', 'atomic')

    _keys = {
        'conv_tol', 'conv_tol_grad', 'max_cycle', 'max_iters',
        'max_stepsize', 'ah_trust_region', 'ah_start_tol',
        'ah_max_cycle', 'init_guess', 'cell', 'mo_coeff', 'kpts'
    }

    def __init__(self, cell, mo_coeff, kpts):
        if isinstance(kpts, KPoints):
            if not kpts.time_reversal:
                raise NotImplementedError('k-point symmetry not implemented')
            mo_coeff = remove_trs_mo(mo_coeff, kpts)
            kpts = kpts.kpts
            logger.warn(cell, 'Time-reversal symmetry will be ignored')

        self.kpts = kpts
        rtypes = [1] + [2] * (len(kpts)-1)  # One gauge-fixed complex rotation
                                            # + (Nk-1) general complex rotations
        kciah.SubspaceCIAHOptimizerMixin.__init__(self, mo_coeff[0].shape[1], rtypes)

        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.mo_coeff = numpy.asarray(mo_coeff)

    def rotate_orb(self, u=None):
        if u is None:
            return self.mo_coeff
        else:
            return numpy.asarray([lib.dot(xk, uk) for xk,uk in zip(self.mo_coeff, u)])

    def dump_flags(self, verbose=None):
        boys.OrbitalLocalizer.dump_flags(self, verbose)

    def get_init_guess(self, key='atomic'):
        ''' Generate initial guess for localization.

            The initial guess is first generated for the first k-point. Other k-points

        Kwargs:
            key : str or bool
                If key is 'atomic', initial guess is based on the projected
                atomic orbitals. False
        '''
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        mo_coeff = self.mo_coeff
        nmo = mo_coeff[0].shape[1]

        mo0 = mo_coeff[0]
        if isinstance(key, str) and key.lower().startswith('atom'):
            u00 = boys.atomic_init_guess(cell, mo0, kpt=kpts[0])
        elif isinstance(key, str) and key.lower().startswith('cho'):
            mo_init = cholesky_mos(mo0)
            S = cell.pbc_intor('int1e_ovlp', kpt=kpts[0])
            u00 = numpy.linalg.multi_dot([mo0.T, S, mo_init])
        else:
            u00 = numpy.eye(nmo)

        # diabatization: align phase of MO[k] to MO[0]
        def align_phase(mo, mo0):
            ovlp = lib.dot(mo.conj().T, mo0)
            l, _, r = numpy.linalg.svd(ovlp)
            return lib.dot(l, r)

        # u0 = self.diabatization(mo_coeff, u00)
        mo0 = lib.dot(mo_coeff[0], u00)
        u0 = [u00]
        for k in range(1,len(mo_coeff)):
            u0.append( align_phase(mo_coeff[k], mo_coeff[0]) )

        return numpy.asarray(u0)

    def get_wannier_function(self, u=None, refcell_only=False):
        mo_coeff = self.rotate_orb(u)

        return wannierization(self.cell, self.kpts, mo_coeff, refcell_only=refcell_only)

    def sort_orb(self, u):
        u = numpy.asarray(u)
        u00 = u.sum(axis=0) / len(self.kpts)
        sorted_idx = mo_mapping.mo_1to1map(u00)
        return self.rotate_orb([uk[:,sorted_idx] for uk in u])

    kernel = boys.kernel


class KptsOrbitalLocalizerReal(KptsOrbitalLocalizer):

    def __init__(self, cell, mo_coeff, kpts):
        if isinstance(kpts, KPoints):
            if not kpts.time_reversal:
                raise NotImplementedError('k-point symmetry not implemented')
            mo_coeff = remove_trs_mo(mo_coeff, kpts)
            kpts_symm = kpts
        else:
            if not gamma_point(kpts[0]):
                raise ValueError('Input k-mesh is not Gamma-centered.')
            kmesh = get_kmesh(cell, kpts)
            kpts_symm = cell.make_kpts(kmesh, time_reversal_symmetry=True)

        self.kpts = kpts_symm.kpts
        self.kpts_symm = kpts_symm

        # Time-reversal invariant k-points => real rotation (type 0)
        # Time-reversal paired    k-points => complex rotation (type 2)
        rtypes = numpy.zeros(kpts_symm.nkpts_ibz, dtype=int)
        for q in range(kpts_symm.nkpts_ibz):
            idx = numpy.where(kpts_symm.bz2ibz==q)[0]
            if idx.size == 2:
                rtypes[q] = 2

        kciah.SubspaceCIAHOptimizerMixin.__init__(self, mo_coeff[0].shape[1], rtypes)

        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.mo_coeff = numpy.asarray(mo_coeff)

    def rotate_orb(self, u=None):
        if u is None:
            return self.mo_coeff
        else:
            mo_coeff = numpy.zeros_like(self.mo_coeff)
            for q in range(self.kpts_symm.nkpts_ibz):
                idx = numpy.where(self.kpts_symm.bz2ibz==q)[0]
                mo_coeff[idx[0]] = numpy.dot(self.mo_coeff[idx[0]], u[q])
                if idx.size == 2:
                    mo_coeff[idx[1]] = numpy.dot(self.mo_coeff[idx[1]], u[q].conj())
            return mo_coeff


class KptsPipekMezey(KptsOrbitalLocalizer):
    '''The Pipek-Mezey localization optimizer that maximizes the orbital
    population

    Args:
        cell : Mole object

    Kwargs:
        mo_coeff : size (N,N) numpy.array
            The orbital space to localize for PM localization.
            When initializing the localization optimizer ``bopt = PM(mo_coeff)``,

            Note these orbitals ``mo_coeff`` may or may not be used as initial
            guess, depending on the attribute ``.init_guess`` . If ``.init_guess``
            is set to None, the ``mo_coeff`` will be used as initial guess. If
            ``.init_guess`` is 'atomic', a few atomic orbitals will be
            constructed inside the space of the input orbitals and the atomic
            orbitals will be used as initial guess.

            Note when calling .kernel(orb) method with a set of orbitals as
            argument, the orbitals will be used as initial guess regardless of
            the value of the attributes .mo_coeff and .init_guess.

    Attributes for PM class:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`.
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`.
        conv_tol : float
            Converge threshold.  Default 1e-6
        conv_tol_grad : float
            Converge threshold for orbital rotation gradients.  Default 1e-3
        max_cycle : int
            The max. number of macro iterations. Default 100
        max_iters : int
            The max. number of iterations in each macro iteration. Default 20
        max_stepsize : float
            The step size for orbital rotation.  Small step (0.005 - 0.05) is preferred.
            Default 0.03.
        init_guess : str or None
            Initial guess for optimization. If set to None, orbitals defined
            by the attribute .mo_coeff will be used as initial guess. If set
            to 'atomic', atomic orbitals will be used as initial guess.
            Default 'atomic'
        pop_method : str
            How the orbital population is calculated, see JCTC 10, 642
            (2014) for discussion. Options are:
            - 'meta-lowdin' (default) as defined in JCTC 10, 3784 (2014)
            - 'mulliken' original Pipek-Mezey scheme, JCP 90, 4916 (1989)
            - 'lowdin' Lowdin charges, JCTC 10, 642 (2014)
            - 'iao' or 'ibo' intrinsic atomic orbitals, JCTC 9, 4384 (2013)
            - 'becke' Becke charges, JCTC 10, 642 (2014)
            The IAO and Becke charges do not depend explicitly on the
            basis set, and have a complete basis set limit [JCTC 10,
            642 (2014)].
        exponent : int
            The power to define norm. It can be 2 or 4. Default 2.

    Saved results

        mo_coeff : ndarray
            Localized orbitals

    '''


    pop_method = getattr(__config__, 'lo_pipek_PM_pop_method', 'meta_lowdin')
    conv_tol = getattr(__config__, 'lo_pipek_PM_conv_tol', 1e-6)
    exponent = getattr(__config__, 'lo_pipek_PM_exponent', 2)  # any integer >= 2

    _keys = {'pop_method', 'conv_tol', 'exponent', '_proj_data'}

    def __init__(self, cell, mo_coeff, kpts, pop_method=None):
        KptsOrbitalLocalizer.__init__(self, cell, mo_coeff, kpts)
        if pop_method is not None:
            self.pop_method = pop_method
        self._proj_data = None

    def dump_flags(self, verbose=None):
        KptsOrbitalLocalizer.dump_flags(self, verbose)
        logger.info(self, 'pop_method = %s',self.pop_method)
        logger.info(self, 'exponent = %s',self.exponent)

    def get_proj_data(self, cell=None, mo_coeff=None, method=None, kpts=None):
        if cell is None: cell = self.cell
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if method is None: method = self.pop_method.lower().replace('_', '-')
        if kpts is None: kpts = self.kpts

        mo_coeff = numpy.asarray(mo_coeff)
        nkpts,nao,nmo = mo_coeff.shape
        kmesh = get_kmesh(cell, kpts)
        scell, phase = k2gamma.get_phase(cell, kpts, kmesh=kmesh)

        if method == 'mulliken':
            proj_data = None

        elif method in ('lowdin', 'meta-lowdin'):
            s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
            proj_coeff = numpy.asarray([orth.orth_ao(cell, method, 'ANO', s=s[k],
                                        adjust_phase=False) for k in range(nkpts)])
            proj_coeff = lib.einsum('kmx,Sk->kmSx', proj_coeff, phase.conj()) / nkpts**0.5
            offset_nr_by_atom = scell.offset_nr_by_atom()
            proj_data = (proj_coeff, s, offset_nr_by_atom)

        elif method in ('iao', 'ibo', 'iao-biorth'):
            s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
            iao_coeff = iao.iao(cell, mo_coeff, kpts=kpts)
            iao_scell = iao.reference_mol(scell)
            offset_nr_by_atom = iao_scell.offset_nr_by_atom()

            if method == 'iao-biorth':
                proj_coeff = lib.einsum('kmx,Sk->kmSx', iao_coeff, phase.conj()) / nkpts**0.5
                ovlp = lib.einsum('kmx,kmn,kny->kxy', iao_coeff.conj(), s, iao_coeff)
                iaotild_coeff = numpy.asarray([numpy.linalg.solve(ovlp[k],
                                               iao_coeff[k].conj().T).conj().T
                                               for k in range(nkpts)], order='C')
                projtild_coeff = lib.einsum('kmx,Sk->kmSx', iaotild_coeff,
                                            phase.conj()) / nkpts**0.5
                proj_data = (proj_coeff, projtild_coeff, s, offset_nr_by_atom)
            else:
                proj_coeff = numpy.asarray([orth.vec_lowdin(iao_coeff[k], s[k])
                                            for k in range(nkpts)])
                proj_coeff = lib.einsum('kmx,Sk->kmSx', proj_coeff, phase.conj()) / nkpts**0.5
                proj_data = (proj_coeff, s, offset_nr_by_atom)

        else:
            raise KeyError('method = %s' % method)

        return proj_data

    # def gen_g_hop(self, u=None):
    #     exponent = self.exponent
    #     proj = self.atomic_pops(u).transpose(1,3,0,2,4)
    #
    #     proj0k = proj.sum(axis=0)
    #     popkk = lib.einsum('kkxii->kxi', proj)
    #     pop0 = lib.einsum('kxii->xi', proj0k.real)
    #     pop0exp1 = pop0**(exponent-1)
    #     pop0exp2 = pop0**(exponent-2)
    #
    #     # gradient
    #     g = self.get_grad(proj0k=proj0k)
    #
    #     # hessian diagonal
    #     g1 = lib.einsum('xi,txij->tij', pop0exp2, proj0k.real**2)
    #     g2 = lib.einsum('xi,txij->tij', pop0exp2, proj0k.imag**2)
    #     h_diag = -4 * exponent * (exponent-1) * (g1 + g2 * 1j)
    #     g1 = lib.einsum('xi,kxii->ki', pop0exp1, proj0k.real)
    #     g2 = lib.einsum('xi,kxj->kij', pop0exp1, popkk.real)
    #     h_diag += 2 * exponent * (g1[:,:,None] - g2) * (1 + 1j)
    #     for hk in h_diag:
    #         numpy.fill_diagonal(hk, numpy.diag(hk)*0.5)
    #         hk += hk.T
    #     h_diag = self.pack_uniq_var(h_diag)
    #
    #     # hessian vector product
    #     # QPkt1 = get_QP(self.cell, self.rotate_orb(u), self.kpts, self.exponent, method=self.pop_method)
    #     QPkt = lib.einsum('xj,ktxil->ktilj', pop0exp1, proj)
    #     Gk = lib.einsum('xi,kxij->kij', pop0exp1, proj0k)
    #
    #     def h_op(x):
    #         x = self.unpack_uniq_var(x)
    #
    #         # contributions from disconnected term
    #         proj0xR = lib.einsum('txil,tlj->xij', proj0k, x).real
    #         j0 = pop0exp2 * lib.einsum('xii->xi', proj0xR)
    #         j1 = lib.einsum('xi,kxij->kij', j0, proj0k)
    #         hx = 4 * exponent * (exponent-1) * j1.astype(numpy.complex128)
    #
    #         # contributions symmetric connected terms
    #         j1 = lib.einsum('ktilj,tlj->kij', QPkt, x)
    #         hx += -2 * exponent * j1
    #
    #         # contributions from asymmetric connected terms
    #         j1 = lib.einsum('kil,klj->kij', Gk, x)
    #         j1 += lib.einsum('kil,klj->kij', x, Gk)
    #         hx += exponent * j1
    #
    #         for hxk in hx:
    #             numpy.fill_diagonal(hxk, numpy.diag(hxk)*0.5)
    #             hxk -= hxk.conj().T
    #
    #         return self.pack_uniq_var(hx)
    #
    #     return g, h_op, h_diag

    def gen_g_hop(self, u=None):
        exponent = self.exponent

        mo_coeff = self.rotate_orb(u)
        QPkt, proj0k, popk = atomic_pops_contract(self.cell, mo_coeff, self.kpts, self.exponent,
                                                  method=self.pop_method, proj_data=self._proj_data)
        proj0k = proj0k.transpose(2,0,1,3)
        popkk = popk.transpose(1,0,2)

        pop0 = lib.einsum('kxii->xi', proj0k.real)
        pop0exp1 = pop0**(exponent-1)
        pop0exp2 = pop0**(exponent-2)

        # gradient
        g = self.get_grad(proj0k=proj0k)

        # hessian diagonal
        g1 = lib.einsum('xi,txij->tij', pop0exp2, proj0k.real**2)
        g2 = lib.einsum('xi,txij->tij', pop0exp2, proj0k.imag**2)
        h_diag = -4 * exponent * (exponent-1) * (g1 + g2 * 1j)
        g1 = lib.einsum('xi,kxii->ki', pop0exp1, proj0k.real)
        g2 = lib.einsum('xi,kxj->kij', pop0exp1, popkk.real)
        h_diag += 2 * exponent * (g1[:,:,None] - g2) * (1 + 1j)
        for hk in h_diag:
            numpy.fill_diagonal(hk, numpy.diag(hk)*0.5)
            hk += hk.T
        h_diag = self.pack_uniq_var(h_diag)

        # hessian vector product
        Gk = lib.einsum('xi,kxij->kij', pop0exp1, proj0k)

        def h_op(x):
            x = self.unpack_uniq_var(x)

            # contributions from disconnected term
            j0 = pop0exp2 * lib.einsum('txil,tli->xi', proj0k, x).real
            j1 = lib.einsum('xi,kxij->kij', j0, proj0k)
            hx = 4 * exponent * (exponent-1) * j1.astype(numpy.complex128)

            # contributions symmetric connected terms
            j1 = lib.einsum('ktilj,tlj->kij', QPkt, x)
            hx += -2 * exponent * j1

            # contributions from asymmetric connected terms
            j1 = lib.einsum('kil,klj->kij', Gk, x)
            j1 += lib.einsum('kil,klj->kij', x, Gk)
            hx += exponent * j1

            for hxk in hx:
                numpy.fill_diagonal(hxk, numpy.diag(hxk)*0.5)
                hxk -= hxk.conj().T

            return self.pack_uniq_var(hx)

        return g, h_op, h_diag

    def get_grad(self, u=None, proj0k=None):
        if proj0k is None:
            proj0k = self.atomic_pops(u, mode='0k').transpose(2,0,1,3)

        exponent = self.exponent

        pop0 = lib.einsum('kxii->xi', proj0k.real)
        pop0exp1 = pop0**(exponent-1)
        g = lib.einsum('xi,kxij->kij', pop0exp1, proj0k)

        for gk in g:
            numpy.fill_diagonal(gk, numpy.diag(gk)*0.5)
            gk -= gk.conj().T

        return 2 * exponent * self.pack_uniq_var(g)

    def cost_function(self, u=None):
        proj00 = self.atomic_pops(u, mode='00')
        return (lib.einsum('xii->xi', proj00.real)**self.exponent).sum()

    @lib.with_doc(atomic_pops.__doc__)
    def atomic_pops(self, u=None, mode='kk'):
        mo_coeff = self.rotate_orb(u)
        proj = atomic_pops(self.cell, mo_coeff, self.kpts, mode=mode, method=self.pop_method,
                           proj_data=self._proj_data)
        return proj

    def kernel(self, mo_coeff=None, callback=None, verbose=None):
        self._proj_data = self.get_proj_data()
        mo_coeff = boys.kernel(self, mo_coeff, callback, verbose)
        self._proj_data = None

        return mo_coeff

    def stability_jacobi(self, verbose=None, return_status=False):
        self._proj_data = self.get_proj_data()
        res = stability_jacobi(self, verbose=verbose, return_status=return_status)
        self._proj_data = None

        return res

    def stability(self, verbose=None, return_status=False):
        self._proj_data = self.get_proj_data()
        res = stability_newton(self, verbose=verbose, return_status=return_status)
        self._proj_data = None

        return res


KPM = KPipek = KptsPipekMezey


class KptsPipekMezeyReal(KptsOrbitalLocalizerReal,KptsPipekMezey):

    def __init__(self, cell, mo_coeff, kpts, pop_method=None):
        KptsOrbitalLocalizerReal.__init__(self, cell, mo_coeff, kpts)
        if pop_method is not None:
            self.pop_method = pop_method

    def gen_g_hop(self, u=None):
        exponent = self.exponent
        proj = self.atomic_pops(u)

        nkpts = len(self.kpts)
        nkpts_ibz = self.kpts_symm.nkpts_ibz
        kmesh = get_kmesh(self.cell, self.kpts)
        scell, phase = k2gamma.get_phase(self.cell, self.kpts, kmesh=kmesh)
        natm = scell.natm

        pop0 = lib.einsum('ktxii->xi', proj.real)
        pop0exp1 = pop0**(exponent-1)
        pop0exp2 = pop0**(exponent-2)
        projkt = proj
        proj0k = lib.einsum('ktxij->txij', projkt)
        popkk = lib.einsum('kkxii->kxi', projkt)

        # gradient
        g = self.get_grad(proj=proj)

        # hessian diagonal
        Proj0k = numpy.zeros((nkpts_ibz,natm,self.norb,self.norb), dtype=numpy.complex)
        Popkk_x = numpy.zeros((nkpts_ibz,natm,self.norb), dtype=numpy.complex)
        Popkk_y = numpy.zeros((nkpts_ibz,natm,self.norb), dtype=numpy.complex)
        for q in range(nkpts_ibz):
            idx = numpy.where(self.kpts_symm.bz2ibz==q)[0]
            if idx.size == 1:
                Proj0k[q] = proj0k[idx[0]]
                Popkk_x[q] = Popkk_y[q] = popkk[idx[0]]
            else:
                Proj0k[q] = proj0k[idx[0]] + proj0k[idx[1]].conj()
                k1, k2 = idx
                Popkk_x[q]  = lib.einsum('xii->xi', projkt[k1,k1])
                Popkk_x[q] += lib.einsum('xii->xi', projkt[k2,k2])
                Popkk_x[q] += lib.einsum('xii->xi', projkt[k1,k2])
                Popkk_x[q] += lib.einsum('xii->xi', projkt[k2,k1])
                Popkk_y[q]  = lib.einsum('xii->xi', projkt[k1,k1])
                Popkk_y[q] += lib.einsum('xii->xi', projkt[k2,k2])
                Popkk_y[q] -= lib.einsum('xii->xi', projkt[k1,k2])
                Popkk_y[q] -= lib.einsum('xii->xi', projkt[k2,k1])
                # Popkk[:,q] = popkk[:,idx[0]] + popkk[:,idx[1]].conj()
        g1 = lib.einsum('xi,txij->tij', pop0exp2, Proj0k.real**2)
        g2 = lib.einsum('xi,txij->tij', pop0exp2, Proj0k.imag**2)
        h_diag = -4 * exponent * (exponent-1) * (g1 + g2 * 1j)
        # g1 = lib.einsum('xi,xkii->ki', pop0exp1, Proj0k.real)
        # g2 = lib.einsum('xi,xkj->kij', pop0exp1, Popkk.real)
        # h_diag += 2 * exponent * (g1[:,:,None] - g2) * (1 + 1j)
        g1 = lib.einsum('xi,kxii->ki', pop0exp1, Proj0k.real)
        g2_x = lib.einsum('xi,kxj->kij', pop0exp1, Popkk_x.real)
        g2_y = lib.einsum('xi,kxj->kij', pop0exp1, Popkk_y.real)
        h_diag += 2 * exponent * (g1[:,:,None] * (1 + 1j) - (g2_x + g2_y*1j))
        for k in range(h_diag.shape[0]):
            numpy.fill_diagonal(h_diag[k], numpy.diag(h_diag[k])*0.5)
            h_diag[k] += h_diag[k].T
        h_diag = self.pack_uniq_var(h_diag)


        # hessian vector product
        Gk = lib.einsum('xi,kxij->kij', pop0exp1, proj0k)

        def h_op(X):
            X = self.unpack_uniq_var(X)

            x = numpy.zeros((self.kpts_symm.nkpts, self.norb, self.norb), dtype=numpy.complex128)
            for q in range(nkpts_ibz):
                idx = numpy.where(self.kpts_symm.bz2ibz==q)[0]
                x[idx[0]] = X[q]
                if idx.size == 2:
                    x[idx[1]]= X[q].conj()

            projx = lib.einsum('ktxil,tlj->kxij', projkt, x)
            proj0xR = lib.einsum('kxij->xij', projx.real)

            # contributions from disconnected term
            j0 = pop0exp2 * lib.einsum('xii->xi', proj0xR)
            j1 = lib.einsum('xi,kxij->kij', j0, proj0k)
            hx = 4 * exponent * (exponent-1) * j1.astype(numpy.complex128)

            # contributions symmetric connected terms
            j1 = lib.einsum('xj,kxij->kij', pop0exp1, projx)
            hx += -2 * exponent * j1

            # contributions from asymmetric connected terms
            j1 = lib.einsum('kil,klj->kij', Gk, x)
            j1 += lib.einsum('kil,klj->kij', x, Gk)
            hx += exponent * j1

            for hxk in hx:
                numpy.fill_diagonal(hxk, numpy.diag(hxk)*0.5)
                hxk -= hxk.conj().T

            HX = numpy.zeros((nkpts_ibz, self.norb, self.norb), dtype=numpy.complex128)
            for q in range(nkpts_ibz):
                idx = numpy.where(self.kpts_symm.bz2ibz==q)[0]
                if idx.size == 1:
                    HX[q] = hx[idx[0]]
                else:
                    HX[q] = hx[idx[0]] + hx[idx[1]].conj()

            return self.pack_uniq_var(HX)

        return g, h_op, h_diag

    def get_grad(self, u=None, proj=None):
        if proj is None:
            proj = self.atomic_pops(u)

        kmesh = get_kmesh(self.cell, self.kpts)
        scell, phase = k2gamma.get_phase(self.cell, self.kpts, kmesh=kmesh)

        exponent = self.exponent

        pop0 = lib.einsum('ktxii->xi', proj.real)
        pop0exp1 = pop0**(exponent-1)
        projkt = proj
        proj0k = lib.einsum('ktxij->txij', projkt)
        g = lib.einsum('xi,kxij->kij', pop0exp1, proj0k)

        for gk in g:
            numpy.fill_diagonal(gk, numpy.diag(gk)*0.5)
            gk -= gk.conj().T

        G = numpy.zeros((self.kpts_symm.nkpts_ibz, self.norb, self.norb), dtype=numpy.complex128)
        for q in range(self.kpts_symm.nkpts_ibz):
            idx = numpy.where(self.kpts_symm.bz2ibz==q)[0]
            if idx.size == 1:
                G[q] = g[idx[0]]
            else:
                G[q] = g[idx[0]] + g[idx[1]].conj()

        return 2 * exponent * self.pack_uniq_var(G)


KPMReal = KPipekReal = KptsPipekMezeyReal


def get_kmesh(cell, kpts, tol=1e-6, nmax=100):
    scaled_kpts = cell.get_scaled_kpts(kpts-kpts[0])
    kmesh = []
    for i in range(3):
        found = False
        for n in range(1,nmax+1):
            ks = scaled_kpts[:,i]*n
            if numpy.all(abs(ks - numpy.round(ks)) < tol):
                found = True
                break
        if not found:
            raise RuntimeError('Input kmesh is either too large or not a (shifted) regular mesh.')
        kmesh.append(n)

    return kmesh

def remove_trs_mo(mo_coeff_trs, kpts):
    assert( len(mo_coeff_trs) == kpts.nkpts_ibz )

    mo_coeff_trs = numpy.asarray(mo_coeff_trs, order='C')
    kpairs = [numpy.where(kpts.bz2ibz==q)[0] for q in range(kpts.nkpts_ibz)]
    mo_coeff = numpy.empty((kpts.nkpts,*mo_coeff_trs[0].shape), dtype=mo_coeff_trs.dtype)

    for q,kpair in enumerate(kpairs):
        if len(kpair) == 1:
            k = kpair[0]
            mo_coeff[k] = mo_coeff_trs[q]
        else:
            k1, k2 = kpair
            mo_coeff[k1] = mo_coeff_trs[q].conj()
            mo_coeff[k2] = mo_coeff_trs[q]

    return mo_coeff

def wannierization(cell, kpts, mo_coeff, kmesh=None, refcell_only=False):
    mo_coeff = numpy.asarray(mo_coeff, order='C')
    if kmesh is None: kmesh = get_kmesh(cell, kpts)
    scell, phase = k2gamma.get_phase(cell, kpts, kmesh=kmesh)
    nkpts,nao,nmo = mo_coeff.shape
    Nao = nkpts*nao

    if refcell_only:
        W = lib.einsum('Rk,kmi,k->Rmi', phase, mo_coeff, phase[0].conj()).reshape(Nao,nmo)
    else:
        W = lib.einsum('Rk,kmi,Sk->RmSi', phase, mo_coeff, phase.conj()).reshape(Nao,nkpts*nmo)

    return W



if __name__ == '__main__':
    # from pyscf.pbc import gto, scf
    #
    # atom = '''
    # O          0.00000        0.00000        0.11779
    # H          0.00000        0.75545       -0.47116
    # H          0.00000       -0.75545       -0.47116
    # '''
    # basis = 'ccpvdz'
    # a = numpy.eye(3) * 5
    # frozen = 1
    #
    # cell = gto.M(atom=atom, a=a, basis=basis).set(verbose=3)
    # nocc = cell.nelectron // 2
    #
    # kpts = cell.make_kpts([2,1,1], scaled_center=[0.37, 0.21, 0.85])
    # # kpts = cell.make_kpts([2,1,1])
    #
    # mf = scf.KRKS(cell, kpts).rs_density_fit()
    # mf.kernel()
    #
    # mo0 = numpy.asarray([x[:,frozen:nocc] for x in mf.mo_coeff])
    # mlo = KPM(cell, mo0, kpts)
    # mlo.verbose = 4
    # mlo.pop_method = 'iao-biorth'
    # mlo.kernel()
    #
    # # stability check
    # while True:
    #     mo, stable = mlo.stability_jacobi(return_status=True)
    #     # mo, stable = mlo.stability(return_status=True)
    #     if stable:
    #         break
    #     mlo.kernel(mo)
    #
    # mlo = PMComplex(cell, mo, kpt=kpt)
    # mlo.kernel()
    #
    # # stability check
    # while True:
    #     mo, stable = mlo.stability_jacobi(return_status=True)
    #     # mo, stable = mlo.stability(return_status=True)
    #     if stable:
    #         break
    #     mlo.kernel(mo)


    from pyscf.pbc import gto, scf

    cell = gto.Cell()
    cell.atom = '''
    O          0.00000        0.00000        0.11779
    H          0.00000        0.75545       -0.47116
    H          0.00000       -0.75545       -0.47116
    '''
    cell.a = numpy.eye(3) * 5
    cell.basis = 'ccpvdz'
    cell.build()

    kmesh = [3,2,1]
    kpts = cell.make_kpts(kmesh)
    # kpts = cell.make_kpts(kmesh, time_reversal_symmetry=True)
    # kpts = cell.make_kpts(kmesh, scaled_center=[0.37, 0.21, 0.85])

    mf = scf.KRHF(cell, kpts).rs_density_fit().run()

    def findiff_grad(func, x, delta=1e-4):
        ''' Finite-difference gradient
        '''
        x = numpy.asarray(x)
        n = x.size
        g = numpy.zeros_like(x)
        for i in range(n):
            dx = numpy.zeros_like(x)
            dx[i] = delta*0.5
            g[i] = (func(x+dx) - func(x-dx)) / delta
        return g

    def semifindiff_hess(fgrad, x, delta=1e-4):
        x = numpy.asarray(x)
        n = x.size
        h = numpy.zeros((n,n), dtype=x.dtype)
        for i in range(n):
            dxi = numpy.zeros_like(x)
            dxi[i] = delta*0.5
            h[i] = (fgrad(x+dxi) - fgrad(x-dxi)) / delta
        h = (h + h.T) * 0.5
        return h

    nocc = cell.nelectron // 2

    numpy.random.seed(47)

    mo0 = numpy.asarray([x[:,:nocc] for x in mf.mo_coeff])
    # mo0 = mo0 + numpy.random.rand(*mo0.shape) * (0.1)
    # mo0 = mo0 + numpy.random.rand(*mo0.shape) * (0.1+0.1j)
    # mlo = PM(cell, mo0)
    mlo = KPM(cell, mo0, kpts)
    # mlo = KPMReal(cell, mo0, kpts)

    wann_coeff = mlo.get_wannier_function()
    print('max|W.imag| = %.15g' % (abs(wann_coeff.imag).max()))

    # mlo = KPMReal(cell, mo0, kpts)
    mlo.verbose = 4
    mlo.init_guess = 'cho'
    # mlo.pop_method = 'iao'
    mlo.kernel()

    # stability check
    while True:
        mo, stable = mlo.stability_jacobi(return_status=True)
        # mo, stable = mlo.stability(return_status=True)
        if stable:
            break
        mlo.kernel(mo)

    wann_coeff = mlo.get_wannier_function()
    print('max|W.imag| = %.15g' % (abs(wann_coeff.imag).max()))

    import sys
    sys.exit(1)

    # finite difference
    nkpts = len(kpts)
    def func(x):
        u = mlo.extract_rotation(x)
        return -mlo.cost_function(u)

    def fgrad(x):
        u = mlo.extract_rotation(x)
        return mlo.get_grad(u)

    g, h_op, h_diag = mlo.gen_g_hop()
    x = mlo.zero_uniq_var()
    h = numpy.zeros((mlo.pdim,mlo.pdim))
    for i in range(mlo.pdim):
        x[i] = 1
        h[:,i] = h_op(x)
        x[i] = 0
    gp = mlo.get_grad()

    g1 = findiff_grad(func, x)
    g_err = abs(g-g1).max()
    print(g)
    print(g1)
    print(f'Grad err: {g_err:.3e}')

    # h1 = findiff_hess(func, x)
    h1 = semifindiff_hess(fgrad, x)
    h_err = abs(h-h1).max()
    print(f'Hess err: {h_err:.3e}')

    hd_err = abs(h_diag-numpy.diag(h1)).max()
    print(h_diag)
    print(numpy.diag(h1))
    print(f'Hess-diag err: {hd_err:.3e}')

    # mlo = PM(cell)
    # mlo.verbose = 4
    # mlo.exponent = 2    # integer >= 2
    # mo0 = mf.mo_coeff[:,mf.mo_occ>1e-6]
    # mo = mlo.kernel(mo0)
    # isstable, mo1 = mlo.stability_jacobi()
    # if not isstable:
    #     mo = mlo.kernel(mo1)
    #     isstable, mo1 = mlo.stability_jacobi()
    #     assert( isstable )
