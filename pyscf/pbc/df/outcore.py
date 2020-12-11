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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import h5py
from pyscf import lib
from pyscf import gto
from pyscf.ao2mo.outcore import balance_segs
from pyscf.pbc.lib.kpts_helper import gamma_point, unique, KPT_DIFF_TOL
from pyscf.pbc.df.incore import wrap_int3c, make_auxcell

libpbc = lib.load_library('libpbc')


def aux_e1(cell, auxcell_or_auxbasis, erifile, intor='int3c2e', aosym='s2ij', comp=None,
           kptij_lst=None, dataname='eri_mo', shls_slice=None, max_memory=2000,
           verbose=0):
    r'''3-center AO integrals (L|ij) with double lattice sum:
    \sum_{lm} (L[0]|i[l]j[m]), where L is the auxiliary basis.
    Three-index integral tensor (kptij_idx, naux, nao_pair) or four-index
    integral tensor (kptij_idx, comp, naux, nao_pair) are stored on disk.

    Args:
        kptij_lst : (*,2,3) array
            A list of (kpti, kptj)
    '''
    if isinstance(auxcell_or_auxbasis, gto.Mole):
        auxcell = auxcell_or_auxbasis
    else:
        auxcell = make_auxcell(cell, auxcell_or_auxbasis)

    intor, comp = gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)

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
        kptij_lst = numpy.zeros((1,2,3))
    feri[dataname+'-kptij'] = kptij_lst

    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)

    ao_loc = cell.ao_loc_nr()
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)[:shls_slice[5]+1]
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    naux = aux_loc[shls_slice[5]] - aux_loc[shls_slice[4]]
    nkptij = len(kptij_lst)

    nii = (ao_loc[shls_slice[1]]*(ao_loc[shls_slice[1]]+1)//2 -
           ao_loc[shls_slice[0]]*(ao_loc[shls_slice[0]]+1)//2)
    nij = ni * nj

    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]
    aosym_ks2 = abs(kpti-kptj).sum(axis=1) < KPT_DIFF_TOL
    j_only = numpy.all(aosym_ks2)
    #aosym_ks2 &= (aosym[:2] == 's2' and shls_slice[:2] == shls_slice[2:4])
    aosym_ks2 &= aosym[:2] == 's2'
    for k, kptij in enumerate(kptij_lst):
        key = '%s/%d' % (dataname, k)
        if gamma_point(kptij):
            dtype = 'f8'
        else:
            dtype = 'c16'
        if aosym_ks2[k]:
            nao_pair = nii
        else:
            nao_pair = nij
        if comp == 1:
            shape = (naux,nao_pair)
        else:
            shape = (comp,naux,nao_pair)
        feri.create_dataset(key, shape, dtype)
    if naux == 0:
        feri.close()
        return erifile

    if j_only and aosym[:2] == 's2':
        assert(shls_slice[2] == 0)
        nao_pair = nii
    else:
        nao_pair = nij

    if gamma_point(kptij_lst):
        dtype = numpy.double
    else:
        dtype = numpy.complex128

    buflen = max(8, int(max_memory*1e6/16/(nkptij*ni*nj*comp)))
    auxdims = aux_loc[shls_slice[4]+1:shls_slice[5]+1] - aux_loc[shls_slice[4]:shls_slice[5]]
    auxranges = balance_segs(auxdims, buflen)
    buflen = max([x[2] for x in auxranges])
    buf = numpy.empty(nkptij*comp*ni*nj*buflen, dtype=dtype)
    buf1 = numpy.empty(ni*nj*buflen, dtype=dtype)

    int3c = wrap_int3c(cell, auxcell, intor, aosym, comp, kptij_lst)

    naux0 = 0
    for istep, auxrange in enumerate(auxranges):
        sh0, sh1, nrow = auxrange
        sub_slice = (shls_slice[0], shls_slice[1],
                     shls_slice[2], shls_slice[3],
                     shls_slice[4]+sh0, shls_slice[4]+sh1)
        mat = numpy.ndarray((nkptij,comp,nao_pair,nrow), dtype=dtype, buffer=buf)
        mat = int3c(sub_slice, mat)

        for k, kptij in enumerate(kptij_lst):
            h5dat = feri['%s/%d'%(dataname,k)]
            for icomp, v in enumerate(mat[k]):
                v = lib.transpose(v, out=buf1)
                if gamma_point(kptij):
                    v = v.real
                if aosym_ks2[k] and v.shape[1] == ni**2:
                    v = lib.pack_tril(v.reshape(-1,ni,ni))
                if comp == 1:
                    h5dat[naux0:naux0+nrow] = v
                else:
                    h5dat[icomp,naux0:naux0+nrow] = v
        naux0 += nrow

    if not isinstance(erifile, h5py.Group):
        feri.close()
    return erifile


def _aux_e2(cell, auxcell_or_auxbasis, erifile, intor='int3c2e', aosym='s2ij', comp=None,
            kptij_lst=None, dataname='eri_mo', shls_slice=None, max_memory=2000,
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
    if isinstance(auxcell_or_auxbasis, gto.Mole):
        auxcell = auxcell_or_auxbasis
    else:
        auxcell = make_auxcell(cell, auxcell_or_auxbasis)

    intor, comp = gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)

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
        kptij_lst = numpy.zeros((1,2,3))
    feri[dataname+'-kptij'] = kptij_lst

    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)

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
    j_only = numpy.all(aosym_ks2)
    #aosym_ks2 &= (aosym[:2] == 's2' and shls_slice[:2] == shls_slice[2:4])
    aosym_ks2 &= aosym[:2] == 's2'

    if j_only and aosym[:2] == 's2':
        assert(shls_slice[2] == 0)
        nao_pair = nii
    else:
        nao_pair = nij

    if gamma_point(kptij_lst):
        dtype = numpy.double
    else:
        dtype = numpy.complex128

    buflen = max(8, int(max_memory*.47e6/16/(nkptij*ni*nj*comp)))
    auxdims = aux_loc[shls_slice[4]+1:shls_slice[5]+1] - aux_loc[shls_slice[4]:shls_slice[5]]
    auxranges = balance_segs(auxdims, buflen)
    buflen = max([x[2] for x in auxranges])
    buf = numpy.empty(nkptij*comp*ni*nj*buflen, dtype=dtype)
    buf1 = numpy.empty_like(buf)

    int3c = wrap_int3c(cell, auxcell, intor, aosym, comp, kptij_lst)

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
# sorted_ij_idx: Sort and group the kptij_lst according to the ordering in
# df._make_j3c to reduce the data fragment in the hdf5 file.  When datasets
# are written to hdf5, they are saved sequentially. If the integral data are
# saved as the order of kptij_lst, removing the datasets in df._make_j3c will
# lead to holes that can not be reused.
    sorted_ij_idx = numpy.hstack([numpy.where(uniq_inverse == k)[0]
                                  for k, kpt in enumerate(uniq_kpts)])
    tril_idx = numpy.tril_indices(ni)
    tril_idx = tril_idx[0] * ni + tril_idx[1]
    def save(istep, mat):
        for k in sorted_ij_idx:
            v = mat[k]
            if gamma_point(kptij_lst[k]):
                v = v.real
            if aosym_ks2[k] and nao_pair == ni**2:
                v = v[:,tril_idx]
            feri['%s/%d/%d' % (dataname,k,istep)] = v

    with lib.call_in_background(save) as bsave:
        for istep, auxrange in enumerate(auxranges):
            sh0, sh1, nrow = auxrange
            sub_slice = (shls_slice[0], shls_slice[1],
                         shls_slice[2], shls_slice[3],
                         shls_slice[4]+sh0, shls_slice[4]+sh1)
            mat = numpy.ndarray((nkptij,comp,nao_pair,nrow), dtype=dtype, buffer=buf)
            bsave(istep, int3c(sub_slice, mat))
            buf, buf1 = buf1, buf

    if not isinstance(erifile, h5py.Group):
        feri.close()
    return erifile


def _aux_e2_hy(cell, auxcell_or_auxbasis, erifile, intor='int3c2e',
               aosym='s2ij', comp=None, kptij_lst=None, dataname='eri_mo',
               shls_slice=None, max_memory=2000,
               bvk_kmesh=None,
               prescreening_type=0, prescreening_data=None,
               cell_fat=None,
               verbose=0):

    if cell_fat is None:
        # _aux_e2_hy_nosplitbas(cell, auxcell_or_auxbasis, erifile, intor=intor,
        _aux_e2_hy_nosplitbas_batchkpt(cell, auxcell_or_auxbasis, erifile, intor=intor,
                              aosym=aosym, comp=comp, kptij_lst=kptij_lst,
                              dataname=dataname, shls_slice=shls_slice,
                              max_memory=max_memory,
                              bvk_kmesh=bvk_kmesh,
                              prescreening_type=prescreening_type,
                              prescreening_data=prescreening_data,
                              verbose=verbose)
    else:
        # _aux_e2_hy_nosplitbas(cell_fat, auxcell_or_auxbasis, erifile,
        #                       intor=intor, aosym=aosym, comp=comp, kptij_lst=kptij_lst,
        #                       dataname=dataname, shls_slice=shls_slice,
        #                       max_memory=max_memory,
        #                       bvk_kmesh=bvk_kmesh,
        #                       prescreening_type=prescreening_type,
        #                       prescreening_data=prescreening_data,
        #                       verbose=verbose)
        # _aux_e2_hy_splitbas(cell, cell_fat, auxcell_or_auxbasis, erifile,
        #                     intor=intor, aosym=aosym, comp=comp,
        #                     kptij_lst=kptij_lst, dataname=dataname,
        #                     shls_slice=shls_slice, max_memory=max_memory,
        #                     bvk_kmesh=bvk_kmesh,
        #                     prescreening_type=prescreening_type,
        #                     prescreening_data=prescreening_data,
        #                     verbose=verbose)
        _aux_e2_hy_splitbas2(cell, cell_fat, auxcell_or_auxbasis, erifile,
                            intor=intor, aosym=aosym, comp=comp,
                            kptij_lst=kptij_lst, dataname=dataname,
                            shls_slice=shls_slice, max_memory=max_memory,
                            bvk_kmesh=bvk_kmesh,
                            prescreening_type=prescreening_type,
                            prescreening_data=prescreening_data,
                            verbose=verbose)


def _aux_e2_hy_nosplitbas(cell, auxcell_or_auxbasis, erifile,
               intor='int3c2e', aosym='s2ij', comp=None, kptij_lst=None,
               dataname='eri_mo', shls_slice=None, max_memory=2000,
               bvk_kmesh=None,
               prescreening_type=0, prescreening_data=None,
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
    if isinstance(auxcell_or_auxbasis, gto.Mole):
        auxcell = auxcell_or_auxbasis
    else:
        auxcell = make_auxcell(cell, auxcell_or_auxbasis)

    intor, comp = gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)

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
        kptij_lst = numpy.zeros((1,2,3))
    feri[dataname+'-kptij'] = kptij_lst

    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)

    shlpr_mask = numpy.ones((shls_slice[1]-shls_slice[0],
                             shls_slice[3]-shls_slice[2]),
                             dtype=numpy.int8, order="C")

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
    j_only = numpy.all(aosym_ks2)
    #aosym_ks2 &= (aosym[:2] == 's2' and shls_slice[:2] == shls_slice[2:4])
    aosym_ks2 &= aosym[:2] == 's2'

    if j_only and aosym[:2] == 's2':
        assert(shls_slice[2] == 0)
        nao_pair = nii
    else:
        nao_pair = nij

    if gamma_point(kptij_lst):
        dtype = numpy.double
    else:
        dtype = numpy.complex128

    buflen = max(8, int(max_memory*.47e6/16/(nkptij*ni*nj*comp)))
    auxdims = aux_loc[shls_slice[4]+1:shls_slice[5]+1] - aux_loc[shls_slice[4]:shls_slice[5]]
    auxranges = balance_segs(auxdims, buflen)
    buflen = max([x[2] for x in auxranges])
    buf = numpy.empty(nkptij*comp*ni*nj*buflen, dtype=dtype)
    buf1 = numpy.empty_like(buf)
    mem_in_MB = buf.size*16/1024**2.
    # TODO: significant performance loss is observed when the size of buf exceeds ~1 GB. This happens in large k-mesh where nkptij is large. Simply reducing buflen to keep buf size < 1 GB is not an option, as for large k-mesh even a buflen as small as < 4 requires > 1 GB memory. One solution is to batch nkptij, too.
    if mem_in_MB > max_memory * 0.5:
        raise RuntimeError("Computing 3c2e integrals requires %.2f MB memory, which exceeds the given maximum memory %.2f MB. Try giving PySCF more memory." % (mem_in_MB*2., max_memory))

    if prescreening_type == 0:
        pbcopt = None
    else:
        from pyscf.pbc.gto import _pbcintor
        import copy
        pcell = copy.copy(cell)
        pcell._atm, pcell._bas, pcell._env = \
                        gto.conc_env(cell._atm, cell._bas, cell._env,
                                     cell._atm, cell._bas, cell._env)
        if prescreening_type == 1:
            pbcopt = _pbcintor.PBCOpt1(pcell).init_rcut_cond(pcell,
                                                             prescreening_data)
        elif prescreening_type == 2:
            pbcopt = _pbcintor.PBCOpt2(pcell).init_rcut_cond(pcell,
                                                             prescreening_data)
    from pyscf.pbc.df.incore import wrap_int3c_hy_nosplitbasis
    int3c = wrap_int3c_hy_nosplitbasis(cell, auxcell, shlpr_mask,
                                       intor, aosym, comp, kptij_lst,
                                       bvk_kmesh=bvk_kmesh,
                                       pbcopt=pbcopt,
                                       prescreening_type=prescreening_type)

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
# sorted_ij_idx: Sort and group the kptij_lst according to the ordering in
# df._make_j3c to reduce the data fragment in the hdf5 file.  When datasets
# are written to hdf5, they are saved sequentially. If the integral data are
# saved as the order of kptij_lst, removing the datasets in df._make_j3c will
# lead to holes that can not be reused.
    sorted_ij_idx = numpy.hstack([numpy.where(uniq_inverse == k)[0]
                                  for k, kpt in enumerate(uniq_kpts)])

    tril_idx = numpy.tril_indices(ni)
    tril_idx = tril_idx[0] * ni + tril_idx[1]
    def save(istep, mat):
        for k in sorted_ij_idx:
            v = mat[k]
            if gamma_point(kptij_lst[k]):
                v = v.real
            if aosym_ks2[k] and nao_pair == ni**2:
                v = v[:,tril_idx]
            feri['%s/%d/%d' % (dataname,k,istep)] = v

    with lib.call_in_background(save) as bsave:
        for istep, auxrange in enumerate(auxranges):
            sh0, sh1, nrow = auxrange
            sub_slice = (shls_slice[0], shls_slice[1],
                         shls_slice[2], shls_slice[3],
                         shls_slice[4]+sh0, shls_slice[4]+sh1)
            mat = numpy.ndarray((nkptij,comp,nao_pair,nrow), dtype=dtype, buffer=buf)
            bsave(istep, int3c(sub_slice, mat))
            buf, buf1 = buf1, buf

    if not isinstance(erifile, h5py.Group):
        feri.close()
    return erifile


def estimate_max_nkptij(nkptij, max_memory_aux, auxdims, n0):

    def negmax_nkptij2memaux(negmax_nkptij):
        max_nkptij = min(-negmax_nkptij, nkptij)
        buflen = max(8, int(max_memory_aux*.47e6/16/(max_nkptij*n0)))
        auxranges = balance_segs(auxdims, buflen)
        buflen = max([x[2] for x in auxranges])
        mem_in_MB = max_nkptij*n0*buflen*16/1024**2.

        return mem_in_MB

    if negmax_nkptij2memaux(-nkptij) < max_memory_aux:
        return nkptij

    max_nkptij = nkptij
    while True:
        max_nkptij = int(max_nkptij*0.5)
        memaux = negmax_nkptij2memaux(-max_nkptij)
        if memaux < max_memory_aux:
            break

    from pyscf.pbc.df.rshdf_helper import _binary_search
    negmax_nkptij_range = [-2*max_nkptij, -max_nkptij]
    negmax_nkptij, memaux = _binary_search(negmax_nkptij2memaux,
                                           *negmax_nkptij_range,
                                           max_memory_aux,
                                           1000)

    return -int(negmax_nkptij)


def _aux_e2_hy_nosplitbas_batchkpt(cell, auxcell_or_auxbasis, erifile,
               intor='int3c2e', aosym='s2ij', comp=None, kptij_lst=None,
               dataname='eri_mo', shls_slice=None, max_memory=2000,
               bvk_kmesh=None,
               prescreening_type=0, prescreening_data=None,
               verbose=0):
    r'''Same as '_aux_e2_hy_nosplitbas', but in addition to batching the auxiliary basis index, the kptij index is also batched so that the buf size is kept in 1gb. This helps the code from being bound by I/O.
    '''
    log = lib.logger.Logger(cell.stdout, cell.verbose)

    if isinstance(auxcell_or_auxbasis, gto.Mole):
        auxcell = auxcell_or_auxbasis
    else:
        auxcell = make_auxcell(cell, auxcell_or_auxbasis)

    intor, comp = gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)

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
        kptij_lst = numpy.zeros((1,2,3))
    feri[dataname+'-kptij'] = kptij_lst

    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)

    shlpr_mask = numpy.ones((shls_slice[1]-shls_slice[0],
                             shls_slice[3]-shls_slice[2]),
                             dtype=numpy.int8, order="C")

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
    j_only = numpy.all(aosym_ks2)
    #aosym_ks2 &= (aosym[:2] == 's2' and shls_slice[:2] == shls_slice[2:4])
    aosym_ks2 &= aosym[:2] == 's2'

    if j_only and aosym[:2] == 's2':
        assert(shls_slice[2] == 0)
        nao_pair = nii
    else:
        nao_pair = nij

    if gamma_point(kptij_lst):
        dtype = numpy.double
    else:
        dtype = numpy.complex128

#########

    if prescreening_type == 0:
        pbcopt = None
    else:
        from pyscf.pbc.gto import _pbcintor
        import copy
        pcell = copy.copy(cell)
        pcell._atm, pcell._bas, pcell._env = \
                        gto.conc_env(cell._atm, cell._bas, cell._env,
                                     cell._atm, cell._bas, cell._env)
        if prescreening_type == 1:
            pbcopt = _pbcintor.PBCOpt1(pcell).init_rcut_cond(pcell,
                                                             prescreening_data)
        elif prescreening_type == 2:
            pbcopt = _pbcintor.PBCOpt2(pcell).init_rcut_cond(pcell,
                                                             prescreening_data)

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kptij_lst.tofile("kptij_lst.dat")
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
# sorted_ij_idx: Sort and group the kptij_lst according to the ordering in
# df._make_j3c to reduce the data fragment in the hdf5 file.  When datasets
# are written to hdf5, they are saved sequentially. If the integral data are
# saved as the order of kptij_lst, removing the datasets in df._make_j3c will
# lead to holes that can not be reused.
    sorted_ij_idx = numpy.hstack([numpy.where(uniq_inverse == k)[0]
                                  for k, kpt in enumerate(uniq_kpts)])

    tril_idx = numpy.tril_indices(ni)
    tril_idx = tril_idx[0] * ni + tril_idx[1]
    def save(istep, mat, kseg0, kseg1):
        for kmat,k in enumerate(sorted_ij_idx[kseg0:kseg1]):
            v = mat[kmat]
            if gamma_point(kptij_lst[k]):
                v = v.real
            if aosym_ks2[k] and nao_pair == ni**2:
                v = v[:,tril_idx]
            feri['%s/%d/%d' % (dataname,k,istep)] = v

    auxdims = aux_loc[shls_slice[4]+1:shls_slice[5]+1] - aux_loc[shls_slice[4]:shls_slice[5]]

    # Given maximally allowed memory for batching the auxiliary basis, determine the MAXIMUM kseg size. Note that batching auxiliary basis does not increase the computational cost, but batching kptij_lst does, and that's why we maximize the kseg size.
    from pyscf import __config__
    MAX_MEMORY_AUX = getattr(__config__, 'pbc_df_rshdf2_RSHDF2_MAX_MEMORY_AUX',
                             1000)
    MAX_NKPTIJ = estimate_max_nkptij(nkptij, MAX_MEMORY_AUX, auxdims,
                                     ni*nj*comp)

    log.debug1("Batching kptij_lst and auxbasis by:\n  "
               "MAX_NKPTIJ= %d  MAX_MEMORY_AUX= %d", MAX_NKPTIJ, MAX_MEMORY_AUX)

    # auxranges and hence buflen must be determined only once for all ksegs to avoid different ksegs having different auxranges.
    max_nkptij = min(MAX_NKPTIJ, nkptij)
    buflen = max(8, int(MAX_MEMORY_AUX*.47e6/16/(max_nkptij*ni*nj*comp)))
    auxranges = balance_segs(auxdims, buflen)
    buflen = max([x[2] for x in auxranges])
    buf = numpy.empty(max_nkptij*comp*ni*nj*buflen, dtype=dtype)
    buf1 = numpy.empty_like(buf)

    mem_in_MB = buf.size*16/1024**2.
    log.debug1("memory used by buf+buf1= %.2f MB", mem_in_MB*2.)
    if mem_in_MB > max_memory * 0.5:
        raise RuntimeError("Computing 3c2e integrals requires %.2f MB memory, which exceeds the given maximum memory %.2f MB. Try giving PySCF more memory." % (mem_in_MB*2., max_memory))

    kptij_loc = numpy.concatenate([numpy.arange(0, nkptij, MAX_NKPTIJ),
                                  [nkptij]])
    kptij_nseg = kptij_loc.size - 1

    log.debug1("kptij_lst is split into %d segments", kptij_nseg)
    log.debug1("auxbasis  is split into %d segments", len(auxranges))

    for ikseg in range(kptij_nseg):
        kseg0, kseg1 = kptij_loc[ikseg:ikseg+2]
        kseglen = kseg1 - kseg0
        kptij_seg = kptij_lst[sorted_ij_idx[kseg0:kseg1]]
        log.debug1("kseg= %d  kseg range= (%d, %d)" % (ikseg, kseg0, kseg1))

        from pyscf.pbc.df.incore import wrap_int3c_hy_nosplitbasis
        int3c = wrap_int3c_hy_nosplitbasis(cell, auxcell, shlpr_mask,
                                           intor, aosym, comp, kptij_seg,
                                           bvk_kmesh=bvk_kmesh,
                                           pbcopt=pbcopt,
                                           prescreening_type=prescreening_type)

        with lib.call_in_background(save) as bsave:
            for istep, auxrange in enumerate(auxranges):
                sh0, sh1, nrow = auxrange
                sub_slice = (shls_slice[0], shls_slice[1],
                             shls_slice[2], shls_slice[3],
                             shls_slice[4]+sh0, shls_slice[4]+sh1)
                mat = numpy.ndarray((kseglen,comp,nao_pair,nrow), dtype=dtype,
                                    buffer=buf)
                bsave(istep, int3c(sub_slice, mat), kseg0, kseg1)
                buf, buf1 = buf1, buf

    if not isinstance(erifile, h5py.Group):
        feri.close()
    return erifile


def _aux_e2_hy_splitbas(cell, cell_fat, auxcell_or_auxbasis, erifile,
               intor='int3c2e', aosym='s2ij', comp=None, kptij_lst=None,
               dataname='eri_mo', shls_slice=None, max_memory=2000,
               bvk_kmesh=None,
               prescreening_type=0, prescreening_data=None,
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
    if isinstance(auxcell_or_auxbasis, gto.Mole):
        auxcell = auxcell_or_auxbasis
    else:
        auxcell = make_auxcell(cell, auxcell_or_auxbasis)

    intor, comp = gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)

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
        kptij_lst = numpy.zeros((1,2,3))
    feri[dataname+'-kptij'] = kptij_lst

    if shls_slice is None:
        shls_slice = (0, cell_fat.nbas, 0, cell_fat.nbas, 0, auxcell.nbas)

    n_compact, n_diffuse = cell_fat._nbas_each_set
    shlpr_mask = numpy.ones((shls_slice[1]-shls_slice[0],
                             shls_slice[3]-shls_slice[2]),
                             dtype=numpy.int8, order="C")
    # shlpr_mask[n_compact:,n_compact:] = 0
    print(shlpr_mask)
    bas_idx = cell_fat._bas_idx
    shlpr_idx = bas_idx[:,None]*cell.nbas + bas_idx
    shlpr_idx = numpy.unique(shlpr_idx[numpy.asarray(shlpr_mask,
                             dtype=numpy.bool)])
    mask_dd = numpy.ones(cell.nbas**2, dtype=numpy.bool)
    mask_dd[shlpr_idx] = False
    print(mask_dd)

    ao_loc = cell_fat.ao_loc_nr()
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)[:shls_slice[5]+1]
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    nkptij = len(kptij_lst)

    nii = (ao_loc[shls_slice[1]]*(ao_loc[shls_slice[1]]+1)//2 -
           ao_loc[shls_slice[0]]*(ao_loc[shls_slice[0]]+1)//2)
    nij = ni * nj

    shls_slice0 = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)
    ao_loc0 = cell.ao_loc_nr()
    ni0 = ao_loc0[shls_slice0[1]] - ao_loc0[shls_slice0[0]]
    nj0 = ao_loc0[shls_slice0[3]] - ao_loc0[shls_slice0[2]]

    nii0 = (ao_loc0[shls_slice0[1]]*(ao_loc0[shls_slice0[1]]+1)//2 -
           ao_loc0[shls_slice0[0]]*(ao_loc0[shls_slice0[0]]+1)//2)
    nij0 = ni0 * nj0

    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]
    aosym_ks2 = abs(kpti-kptj).sum(axis=1) < KPT_DIFF_TOL
    j_only = numpy.all(aosym_ks2)
    #aosym_ks2 &= (aosym[:2] == 's2' and shls_slice[:2] == shls_slice[2:4])
    aosym_ks2 &= aosym[:2] == 's2'

    if j_only and aosym[:2] == 's2':
        assert(shls_slice[2] == 0)
        nao_pair = nii
        nao_pair0 = nii0
    else:
        nao_pair = nij
        nao_pair0 = nij0

    if gamma_point(kptij_lst):
        dtype = numpy.double
    else:
        dtype = numpy.complex128

    buflen = max(8, int(max_memory*.47e6/16/(nkptij*ni0*nj0*comp)))
    auxdims = aux_loc[shls_slice[4]+1:shls_slice[5]+1] - aux_loc[shls_slice[4]:shls_slice[5]]
    auxranges = balance_segs(auxdims, buflen)
    buflen = max([x[2] for x in auxranges])
    buf = numpy.empty(nkptij*comp*ni0*nj0*buflen, dtype=dtype)
    buf1 = numpy.empty_like(buf)

    if prescreening_type == 0:
        pbcopt = None
    else:
        from pyscf.pbc.gto import _pbcintor
        import copy
        pcell = copy.copy(cell_fat)
        pcell._atm, pcell._bas, pcell._env = \
                    gto.conc_env(cell_fat._atm, cell_fat._bas, cell_fat._env,
                                 cell_fat._atm, cell_fat._bas, cell_fat._env)
        if prescreening_type == 1:
            pbcopt = _pbcintor.PBCOpt1(pcell).init_rcut_cond(pcell,
                                                             prescreening_data)
        elif prescreening_type == 2:
            pbcopt = _pbcintor.PBCOpt2(pcell).init_rcut_cond(pcell,
                                                             prescreening_data)
    from pyscf.pbc.df.incore import wrap_int3c_hy_splitbasis
    int3c = wrap_int3c_hy_splitbasis(cell_fat, cell, auxcell, shlpr_mask,
                                     intor, aosym, comp, kptij_lst,
                                     bvk_kmesh=bvk_kmesh,
                                     pbcopt=pbcopt,
                                     prescreening_type=prescreening_type)

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
# sorted_ij_idx: Sort and group the kptij_lst according to the ordering in
# df._make_j3c to reduce the data fragment in the hdf5 file.  When datasets
# are written to hdf5, they are saved sequentially. If the integral data are
# saved as the order of kptij_lst, removing the datasets in df._make_j3c will
# lead to holes that can not be reused.
    sorted_ij_idx = numpy.hstack([numpy.where(uniq_inverse == k)[0]
                                  for k, kpt in enumerate(uniq_kpts)])

    tril_idx = numpy.tril_indices(ni0)
    tril_idx = tril_idx[0] * ni0 + tril_idx[1]
    def save(istep, mat):
        mat[:,:,mask_dd] = 0.
        for k in sorted_ij_idx:
            v = mat[k]
            if gamma_point(kptij_lst[k]):
                v = v.real
            if aosym_ks2[k] and nao_pair0 == ni0**2:
                v = v[:,tril_idx]
            feri['%s/%d/%d' % (dataname,k,istep)] = v

    with lib.call_in_background(save) as bsave:
        for istep, auxrange in enumerate(auxranges):
            sh0, sh1, nrow = auxrange
            sub_slice = (shls_slice[0], shls_slice[1],
                         shls_slice[2], shls_slice[3],
                         shls_slice[4]+sh0, shls_slice[4]+sh1)
            mat = numpy.ndarray((nkptij,comp,nao_pair0,nrow), dtype=dtype, buffer=buf)
            bsave(istep, int3c(sub_slice, mat))
            buf, buf1 = buf1, buf

    if not isinstance(erifile, h5py.Group):
        feri.close()
    return erifile


def _aux_e2_hy_splitbas2(cell, cell_fat, auxcell_or_auxbasis, erifile,
               intor='int3c2e', aosym='s2ij', comp=None, kptij_lst=None,
               dataname='eri_mo', shls_slice=None, max_memory=2000,
               bvk_kmesh=None,
               prescreening_type=0, prescreening_data=None,
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
    if isinstance(auxcell_or_auxbasis, gto.Mole):
        auxcell = auxcell_or_auxbasis
    else:
        auxcell = make_auxcell(cell, auxcell_or_auxbasis)

    intor, comp = gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)

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
        kptij_lst = numpy.zeros((1,2,3))
    feri[dataname+'-kptij'] = kptij_lst

    if shls_slice is None:
        shls_slice = (0, cell_fat.nbas, 0, cell_fat.nbas, 0, auxcell.nbas)

    n_compact, n_diffuse = cell_fat._nbas_each_set
    shlpr_mask = numpy.ones((shls_slice[1]-shls_slice[0],
                             shls_slice[3]-shls_slice[2]),
                             dtype=numpy.int8, order="C")
    shlpr_mask[n_compact:,n_compact:] = 0
    print(shlpr_mask)

    ao_loc = cell_fat.ao_loc_nr()
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)[:shls_slice[5]+1]
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    nkptij = len(kptij_lst)

    nii = (ao_loc[shls_slice[1]]*(ao_loc[shls_slice[1]]+1)//2 -
           ao_loc[shls_slice[0]]*(ao_loc[shls_slice[0]]+1)//2)
    nij = ni * nj

    shls_slice0 = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)
    ao_loc0 = cell.ao_loc_nr()
    ni0 = ao_loc0[shls_slice0[1]] - ao_loc0[shls_slice0[0]]
    nj0 = ao_loc0[shls_slice0[3]] - ao_loc0[shls_slice0[2]]

    nii0 = (ao_loc0[shls_slice0[1]]*(ao_loc0[shls_slice0[1]]+1)//2 -
           ao_loc0[shls_slice0[0]]*(ao_loc0[shls_slice0[0]]+1)//2)
    nij0 = ni0 * nj0

    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]
    aosym_ks2 = abs(kpti-kptj).sum(axis=1) < KPT_DIFF_TOL
    j_only = numpy.all(aosym_ks2)
    #aosym_ks2 &= (aosym[:2] == 's2' and shls_slice[:2] == shls_slice[2:4])
    aosym_ks2 &= aosym[:2] == 's2'

    if j_only and aosym[:2] == 's2':
        assert(shls_slice[2] == 0)
        nao_pair = nii
        nao_pair0 = nii0
    else:
        nao_pair = nij
        nao_pair0 = nij0

    if gamma_point(kptij_lst):
        dtype = numpy.double
    else:
        dtype = numpy.complex128

    buflen = max(8, int(max_memory*.47e6/16/(nkptij*ni*nj*comp)))
    auxdims = aux_loc[shls_slice[4]+1:shls_slice[5]+1] - aux_loc[shls_slice[4]:shls_slice[5]]
    auxranges = balance_segs(auxdims, buflen)
    buflen = max([x[2] for x in auxranges])
    buf = numpy.empty(nkptij*comp*ni*nj*buflen, dtype=dtype)
    buf1 = numpy.empty_like(buf)

    # for fat2orig
    buf0 = numpy.empty(nkptij*comp*ni0*nj0*buflen, dtype=dtype)
    buf10 = numpy.empty_like(buf0)

    if prescreening_type == 0:
        pbcopt = None
    else:
        from pyscf.pbc.gto import _pbcintor
        import copy
        pcell = copy.copy(cell_fat)
        pcell._atm, pcell._bas, pcell._env = \
                    gto.conc_env(cell_fat._atm, cell_fat._bas, cell_fat._env,
                                 cell_fat._atm, cell_fat._bas, cell_fat._env)
        if prescreening_type == 1:
            pbcopt = _pbcintor.PBCOpt1(pcell).init_rcut_cond(pcell,
                                                             prescreening_data)
        elif prescreening_type == 2:
            pbcopt = _pbcintor.PBCOpt2(pcell).init_rcut_cond(pcell,
                                                             prescreening_data)
    from pyscf.pbc.df.incore import wrap_int3c_hy_nosplitbasis
    int3c = wrap_int3c_hy_nosplitbasis(cell_fat, auxcell, shlpr_mask,
                                       intor, aosym, comp, kptij_lst,
                                       bvk_kmesh=bvk_kmesh,
                                       pbcopt=pbcopt,
                                       prescreening_type=prescreening_type)

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
# sorted_ij_idx: Sort and group the kptij_lst according to the ordering in
# df._make_j3c to reduce the data fragment in the hdf5 file.  When datasets
# are written to hdf5, they are saved sequentially. If the integral data are
# saved as the order of kptij_lst, removing the datasets in df._make_j3c will
# lead to holes that can not be reused.
    sorted_ij_idx = numpy.hstack([numpy.where(uniq_inverse == k)[0]
                                  for k, kpt in enumerate(uniq_kpts)])

    bas_idx = cell_fat._bas_idx
    def fat2orig(v_fat, v_orig):
        from pyscf.pbc.df.rshdf_ao2mo import fat_orig_loop
        nao0 = ao_loc0[-1]
        aosym_ = 's2' if v_fat.shape[2] == nii else 's1'
        v_orig.fill(0)
        if aosym_ == 's2':
            for iap_fat, iap, iap_fat2 in fat_orig_loop(cell_fat, cell,
                                                        aosym='s2',
                                                        sr_only=True):
                v_orig[:,:,iap] += v_fat[:,:,iap_fat]
                if not iap_fat2 is None:
                    v_orig[:,:,iap] += v_fat[:,:,iap_fat2]
        else:
            for iap_fat, iap in fat_orig_loop(cell_fat, cell, aosym='s1',
                                              sr_only=True):
                v_orig[:,:,iap] += v_fat[:,:,iap_fat]

    tril_idx = numpy.tril_indices(ni0)
    tril_idx = tril_idx[0] * ni0 + tril_idx[1]
    def save(istep, mat_fat, mat_orig):
        fat2orig(mat_fat, mat_orig)
        for k in sorted_ij_idx:
            v = mat_orig[k]
            if gamma_point(kptij_lst[k]):
                v = v.real
            if aosym_ks2[k] and nao_pair0 == ni0**2:
                v = v[:,tril_idx]
            feri['%s/%d/%d' % (dataname,k,istep)] = v

    with lib.call_in_background(save) as bsave:
        for istep, auxrange in enumerate(auxranges):
            sh0, sh1, nrow = auxrange
            sub_slice = (shls_slice[0], shls_slice[1],
                         shls_slice[2], shls_slice[3],
                         shls_slice[4]+sh0, shls_slice[4]+sh1)
            mat = numpy.ndarray((nkptij,comp,nao_pair,nrow), dtype=dtype, buffer=buf)
            mat0 = numpy.ndarray((nkptij,comp,nao_pair0,nrow), dtype=dtype, buffer=buf0)
            bsave(istep, int3c(sub_slice, mat), mat0)
            buf, buf1 = buf1, buf
            buf0, buf10 = buf10, buf0

    if not isinstance(erifile, h5py.Group):
        feri.close()
    return erifile
