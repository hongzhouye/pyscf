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

'''
density fitting MP2,  3-center integrals incore.
'''

import numpy as np
import h5py
import tempfile
import ctypes
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df
from pyscf.mp import mp2
from pyscf.mp.mp2 import make_rdm1, make_rdm2
from pyscf import __config__

einsum = lib.einsum

libmp = lib.load_library('libmp')

WITH_T2 = getattr(__config__, 'mp_dfmp2_with_t2', True)


def kernel(mp, mo_energy, mo_coeff, eris=None, with_t2=WITH_T2, verbose=None):
    """Computes DF-RMP2 energy.

    Args:
        mp (MP2): an instance of MP2
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
        fkernel = kernel_df if with_t2 else kernel_df_C
    elif callable(mp._kernel):
        fkernel = mp._kernel
    elif isinstance(mp._kernel, str):
        fkernel = kernel_df_C if mp._kernel.lower() == 'c' else kernel_df
    else:
        log.error('Unknown kernel type')
        raise ValueError

    return fkernel(mp, mo_energy, mo_coeff, eris, with_t2, verbose)

def kernel_df(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, verbose=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mp)
    log.debug('Using DF-Python kernel')

    if mo_energy is not None or mo_coeff is not None:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:      eris = mp.ao2mo(mo_coeff, with_t2)
    if mo_energy is None: mo_energy = eris.mo_energy
    if mo_coeff is None:  mo_coeff = eris.mo_coeff

    dtype = np.result_type(eris.dtype, mo_coeff.dtype)
    assert(dtype == np.float64)
    dsize = 8
    nocc = mp.nocc
    nvir = mp.nmo - nocc
    naux = mp.with_df.get_naoaux()
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    if with_t2:
        t2 = np.empty((nocc,nocc,nvir,nvir), dtype=dtype)
    else:
        t2 = None

    # determine occ blksize
    mem_avail = mp.max_memory - lib.current_memory()[0]
    # 4*[O]^2*V^2 + 2*[O]XV = mem
    occ_blksize = min(nocc, max(1, int(np.floor(((naux**2+0.8*mem_avail*4*1e6/dsize)**0.5 -
                                                    naux) / (4*nvir)))))
    log.debug('occ blksize for %s loop: %d/%d', mp.__class__.__name__, occ_blksize, nocc)

    cput1 = (logger.process_clock(), logger.perf_counter())

    emp2_ss = emp2_os = 0
    for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc,occ_blksize)):
        iaL = eris.get_ovL(i0,i1)
        for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc,occ_blksize)):
            if ibatch == jbatch:
                jbL = iaL
            else:
                jbL = eris.get_ovL(j0,j1)

            gij = einsum('iaL,jbL->iajb', iaL, jbL)
            t2ij = np.conj(gij) / lib.direct_sum('ia+jb->iajb', eia[i0:i1], eia[j0:j1])
            if with_t2:
                t2[i0:i1,j0:j1] = t2ij.transpose(0,2,1,3)

            ed =  einsum('iajb,iajb->', t2ij, gij)
            ex = -einsum('iajb,ibja->', t2ij, gij)
            emp2_ss += ed + ex
            emp2_os += ed

            t2ij = gij = jbL = None
        iaL = None

        cput1 = log.timer_debug1('i-block [%d:%d]/%d' % (i0,i1,nocc), *cput1)

    log.timer(mp.__class__.__name__, *cput0)

    emp2_ss = emp2_ss
    emp2_os = emp2_os
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2, t2

def kernel_df_C(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, verbose=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mp)
    log.debug('Using DF-C kernel')

    if mo_energy is not None or mo_coeff is not None:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:      eris = mp.ao2mo(mo_coeff, with_t2)
    if mo_energy is None: mo_energy = eris.mo_energy
    if mo_coeff is None:  mo_coeff = eris.mo_coeff

    dtype = np.result_type(eris.dtype, mo_coeff.dtype)
    assert(dtype == np.float64)
    dsize = 8
    nocc = mp.nocc
    nvir = mp.nmo - nocc
    naux = mp.with_df.get_naoaux()

    moeoo = mo_energy[:nocc,None] + mo_energy[:nocc]
    moevv = lib.asarray(mo_energy[nocc:,None] + mo_energy[nocc:], order='C')

    if with_t2:
        raise NotImplementedError
        t2 = np.empty((nocc,nocc,nvir,nvir), dtype=dtype)
    else:
        t2 = None

    drv = libmp.MP2_contract_d

    # determine occ blksize
    mem_avail = mp.max_memory - lib.current_memory()[0]
    # 4*[O]^2*V^2 + 2*[O]XV = mem
    occ_blksize = min(nocc, max(1, int(np.floor(((naux**2+0.8*mem_avail*4*1e6/dsize)**0.5 -
                                                    naux) / (4*nvir)))))
    log.debug('occ blksize for %s loop: %d/%d', mp.__class__.__name__, occ_blksize, nocc)

    cput1 = (logger.process_clock(), logger.perf_counter())

    emp2_ss = emp2_os = 0
    for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc,occ_blksize)):
        nocci = i1-i0
        iaL = eris.get_ovL(i0,i1)
        for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc,occ_blksize)):
            noccj = j1-j0
            if ibatch == jbatch:
                jbL = iaL
            else:
                jbL = eris.get_ovL(j0,j1)

            ed = np.zeros(1, dtype=np.float64)
            ex = np.zeros(1, dtype=np.float64)
            s2symm = 1
            drv(
                ed.ctypes.data_as(ctypes.c_void_p),
                ex.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(s2symm),
                iaL.ctypes.data_as(ctypes.c_void_p),
                jbL.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(i0), ctypes.c_int(j0),
                ctypes.c_int(nocci), ctypes.c_int(noccj),
                ctypes.c_int(nvir), ctypes.c_int(naux),
                lib.asarray(moeoo[i0:i1,j0:j1],
                            order='C').ctypes.data_as(ctypes.c_void_p),
                moevv.ctypes.data_as(ctypes.c_void_p),
            )
            emp2_ss += ed + ex
            emp2_os += ed

            jbL = None
        iaL = None

        cput1 = log.timer_debug1('i-block [%d:%d]/%d' % (i0,i1,nocc), *cput1)

    log.timer(mp.__class__.__name__, *cput0)

    emp2_ss = emp2_ss.real
    emp2_os = emp2_os.real
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2, t2

def _iterative_kernel(mp, eris):
    raise NotImplementedError


class DFMP2(mp2.MP2):

    _kernel = getattr(__config__, 'mp_DFMP2_kernel', None)

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        mp2.MP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
        self._keys.update(['with_df'])

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

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return mp2.MP2.reset(self, mol)

    def ao2mo(self, mo_coeff=None, with_t2=WITH_T2):
        return _make_df_eris(self, mo_coeff, with_t2)

    def make_rdm1(self, t2=None, ao_repr=False):
        if t2 is None:
            t2 = self.t2
        assert t2 is not None
        return make_rdm1(self, t2, ao_repr=ao_repr)

    def make_rdm2(self, t2=None, ao_repr=False):
        if t2 is None:
            t2 = self.t2
        assert t2 is not None
        return make_rdm2(self, t2, ao_repr=ao_repr)

    def nuc_grad_method(self):
        raise NotImplementedError

    # For non-canonical MP2
    def update_amps(self, t2, eris):
        raise NotImplementedError

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

def _mem_usage(nocc, nvir, naux, dsize, with_t2=WITH_T2):
    nmo = nocc + nvir
    basic = nvir**2*4   # vijab, vijba, tijab, eijab
    if with_t2:
        basic += (nocc*nvir)**2
    basic *= dsize / 1e6
    incore = nocc*nvir*naux*dsize / 1e6 + basic
    outcore = basic
    return incore, outcore, basic

class _ChemistsERIs(mp2._ChemistsERIs):
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None

        self.dtype = None
        self.ovL = None
        self._ovL = None
        self._ovL_to_save = None

    def get_ovL(self, i0, i1):
        return np.asarray(self.ovL[i0:i1], order='C')

def _make_df_eris(mymp, mo_coeff=None, with_t2=WITH_T2):
    log = logger.new_logger(mymp)
    time0 = (logger.process_clock(), logger.perf_counter())

    eris = _ChemistsERIs()
    eris._common_init_(mymp, mo_coeff)

    if mo_coeff is None: mo_coeff = eris.mo_coeff

    dtype = mo_coeff.dtype
    assert(dtype == np.float64)
    dsize = 8

    with_df = mymp.with_df

    nocc = mymp.nocc
    nmo = mymp.nmo
    nvir = nmo - nocc
    naux = with_df.get_naoaux()
    mem_incore, mem_outcore, mem_basic = _mem_usage(nocc, nvir, naux, dsize, with_t2)
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
            eris.ovL = np.ndarray((nocc,nvir,naux), dtype=np.float64)
            log.debug('Transformed 3c integrals will be saved in memory')
        else:
            if eris._ovL_to_save is None:
                eris._ovL_to_save = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            if isinstance(eris._ovL_to_save, str):
                eris._ovL = h5py.File(eris._ovL_to_save, 'w')
                log.debug('Transformed 3c integrals will be saved in %s', eris._ovL_to_save)
            else:
                eris._ovL = h5py.File(eris._ovL_to_save.name, 'w')
                log.debug('Transformed 3c integrals will be saved in %s', eris._ovL_to_save.name)
            eris.ovL = eris._ovL.create_dataset('ovL', shape=(nocc,nvir,naux), dtype=dtype)

    _init_mp_df_eris(mymp, mo_coeff, nocc, eris.ovL)

    log.timer('%s ao2mo'%(mymp.__class__.__name__), *time0)

    return eris

def _init_mp_df_eris(mymp, mo_coeff, nocc, ovL=None):
    from pyscf.ao2mo import _ao2mo

    log = logger.Logger(mymp.stdout, mymp.verbose)

    with_df = mymp.with_df

    nmo = mo_coeff.shape[1]
    nvir = nmo - nocc
    nao = mymp.mol.nao_nr()
    nao_pair = nao*(nao+1) // 2
    naux = with_df.get_naoaux()

    dtype = mo_coeff.dtype
    dsize = 8

    mo = np.asarray(mo_coeff, order='F')
    ijslice = (0, nocc, nocc, nmo)

    if ovL is None:
        ovL = np.empty((nocc,nvir,naux), dtype=dtype)

    mem_avail = mymp.max_memory - lib.current_memory()[0]

    if isinstance(ovL, np.ndarray):
        # incore: batching aux (OV + Nao_pair) * [X] = M
        mem_auxblk = (nao_pair+nocc*nvir) * dsize/1e6
        aux_blksize = min(naux, max(1, int(np.floor(mem_avail*0.7 / mem_auxblk))))
        log.debug('aux blksize for incore ao2mo: %d/%d', aux_blksize, naux)
        buf = np.empty(aux_blksize*nocc*nvir, dtype=dtype)
        ijslice = (0,nocc,nocc,nmo)

        p1 = 0
        for Lpq in with_df.loop(blksize=aux_blksize):
            p0, p1 = p1, p1+Lpq.shape[0]
            out = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s2', out=buf)
            ovL[:,:,p0:p1] = out.reshape(-1,nocc,nvir).transpose(1,2,0)
            Lpq = out = None
        buf = None
    else:
        # outcore: batching occ [O]XV and aux ([O]V + Nao_pair)*[X]
        mem_occblk = naux*nvir * dsize/1e6
        occ_blksize = min(nocc, max(1, int(np.floor(mem_avail*0.6 / mem_occblk))))
        mem_auxblk = (occ_blksize*nvir+nao_pair) * dsize/1e6
        aux_blksize = min(naux, max(1, int(np.floor(mem_avail*0.3 / mem_auxblk))))
        log.debug('occ blksize for outcore ao2mo: %d/%d', occ_blksize, nocc)
        log.debug('aux blksize for outcore ao2mo: %d/%d', aux_blksize, naux)
        buf = np.empty(naux*occ_blksize*nvir, dtype=dtype)
        buf2 = np.empty(aux_blksize*occ_blksize*nvir, dtype=dtype)

        for i0,i1 in lib.prange(0,nocc,occ_blksize):
            nocci = i1-i0
            ijslice = (i0,i1,nocc,nmo)
            p1 = 0
            OvL = np.ndarray((nocci,nvir,naux), dtype=dtype, buffer=buf)
            for Lpq in with_df.loop(blksize=aux_blksize):
                p0, p1 = p1, p1+Lpq.shape[0]
                out = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s2', out=buf2)
                OvL[:,:,p0:p1] = out.reshape(-1,nocci,nvir).transpose(1,2,0)
                Lpq = out = None
            ovL[i0:i1] = OvL    # this avoids slow operations like ovL[i0:i1,:,p0:p1] = ...
            OvL = None
        buf = buf2 = None

    return ovL


MP2 = DFMP2

from pyscf import scf
scf.hf.RHF.DFMP2 = lib.class_as_method(DFMP2)
scf.rohf.ROHF.DFMP2 = None
# scf.uhf.UHF.DFMP2 = None

del (WITH_T2)


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.build()
    mf = scf.RHF(mol).run()
    pt = DFMP2(mf)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.204004830285)

    pt.with_df = df.DF(mol)
    pt.with_df.auxbasis = 'weigend'
    emp2, t2 = pt.kernel()
    print(emp2 - -0.204254500453)

    mf = scf.density_fit(scf.RHF(mol), 'weigend')
    mf.kernel()
    pt = DFMP2(mf)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.203986171133)

    pt.with_df = df.DF(mol)
    pt.with_df.auxbasis = df.make_auxbasis(mol, mp2fit=True)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.203738031827)

    pt.frozen = 2
    pt.verbose = 6
    emp2, t2 = pt.kernel()
    print(emp2 - -0.14433975122418313)

    pt.frozen = 2
    pt.verbose = 6
    pt._kernel = 'C'
    emp2, t2 = pt.kernel(with_t2=False)
    print(emp2 - -0.14433975122418313)
