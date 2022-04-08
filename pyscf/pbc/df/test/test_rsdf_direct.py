# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

''' Check if loop_j3c function gives the same j3c as the normal integral-indirect approach
'''

import unittest
import h5py
import tempfile
import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.pbc import gto, df


atom = 'He 0 0 0; He 1 0 0'
a = np.eye(3) * 3
basis = 'cc-pvdz'

cell = gto.Cell(atom=atom, basis=basis, a=a)
cell.build()
cell.verbose = 0

scaled_center0 = np.zeros(3)
scaled_center1 = np.array([0.65881329, 0.40465128, 0.85241511])


def test_gen(kmesh, scaled_center, aosym, j3c_order, partition_iorj):
    nao = cell.nao_nr()
    kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)

    mydf = df.RSDF(cell, kpts)
    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    mydf._cderi_to_save = swapfile.name
    swapfile = None
    mydf.build()

    with h5py.File(mydf._cderi_to_save, 'r') as f:
        kptijs = np.sort(list(map(int,list(f['j3c']))))
        j3c_ref = [f[f'j3c/{k}/0'][()] for k in kptijs]

    mydf2 = df.RSDF(cell, kpts)
    mydf2.direct = True
    mydf2.build()
    naoaux = mydf2.auxcell.nao_nr()

    from pyscf.pbc.df.rsdf_direct_helper import (loop_j3c, get_kptij_lst, get_j2c,
                                                 loop_uniq_q, cholesky_decomposed_metric)
    from pyscf.df.outcore import _guess_shell_ranges
    from pyscf.pbc.lib.kpts_helper import (is_zero, gamma_point, member, unique,
                                           KPT_DIFF_TOL)

    kptij_lst = get_kptij_lst(kpts)
    nkptij = len(kptij_lst)
    uniq_q_loop = [x for x in loop_uniq_q(mydf, kptij_lst=kptij_lst, verbose=0)]
    uniq_kpts = np.asarray([x[0] for x in uniq_q_loop])
    nkpts_uniq = len(uniq_kpts)

    dtype = np.double if is_zero(kptij_lst) else np.complex128
    if dtype == np.double and aosym[:2] == 's2':
        nao_pair = nao*(nao+1)//2
    else:
        nao_pair = nao*nao
    j3c = np.zeros((nkptij,1,naoaux,nao_pair), dtype=dtype)

    kj2c = get_j2c(mydf, kpts=uniq_kpts)
    kj2c_negative = [None] * nkpts_uniq
    kj2ctag = [None] * nkpts_uniq
    for k,kpt in enumerate(uniq_kpts):
        kj2c[k], kj2c_negative[k], kj2ctag[k] = cholesky_decomposed_metric(mydf, kj2c[k])

    blksize = nao_pair // 3
    shranges = _guess_shell_ranges(mydf.cell, blksize, aosym)
    verbose_loop = cell.verbose - 2
    p1 = 0
    for j3cblk in loop_j3c(mydf, kptij_lst=kptij_lst, aosym=aosym, j3c_order=j3c_order,
                           partition_iorj=partition_iorj, shranges=shranges,
                           verbose=verbose_loop):
        if j3c_order == 'ijL':
            j3cblk = j3cblk.transpose(0,1,3,2)
        ncol = j3cblk.shape[-1]
        p0 = p1
        p1 = p0 + ncol
        j3c[:,:,:,p0:p1] = j3cblk

    kq = 0
    errs = []
    for kpt,adapted_kptjs,adapted_ji_idx in uniq_q_loop:
        j2c = kj2c[kq]
        j2ctag = kj2ctag[kq]
        for kptj,ji in zip(adapted_kptjs,adapted_ji_idx):
            j3c_ = scipy.linalg.solve_triangular(j2c, j3c[ji,0], lower=True)
            j3c_ref_ = j3c_ref[ji]
            if j3c_.shape[-1] != j3c_ref_.shape[-1]:
                j3c_ref_ = lib.unpack_tril(j3c_ref_).reshape(naoaux,-1)
            err = abs(j3c_ref_ - j3c_).max()
            errs.append( err )
        kq += 1

    return np.asarray(errs)


class KnownValues(unittest.TestCase):
    def test_j3c_kpt_unshifted(self):
        ''' Gamma
        '''
        kmesh = [1,1,1]
        scaled_center = scaled_center0
        aosym = 's1'
        partition_iorj = 'i'
        for j3c_order in ['Lij','ijL']:
            errs = test_gen(kmesh, scaled_center, aosym, j3c_order, partition_iorj)
            err = np.max(errs)
            self.assertAlmostEqual(err, 0., 10)

    def test_j3c_kpt_shifted(self):
        ''' Single twisted angle
        '''
        kmesh = [1,1,1]
        scaled_center = scaled_center1
        aosym = 's1'
        partition_iorj = 'i'
        for j3c_order in ['Lij','ijL']:
            errs = test_gen(kmesh, scaled_center, aosym, j3c_order, partition_iorj)
            err = np.max(errs)
            self.assertAlmostEqual(err, 0., 10)

    def test_j3c_kpts_unshifted(self):
        kmesh = [3,2,1]
        scaled_center = scaled_center0
        aosym = 's1'
        partition_iorj = 'i'
        for j3c_order in ['Lij','ijL']:
            errs = test_gen(kmesh, scaled_center, aosym, j3c_order, partition_iorj)
            err = np.max(errs)
            self.assertAlmostEqual(err, 0., 10)

    def test_j3c_kpts_shifted(self):
        kmesh = [3,2,1]
        scaled_center = scaled_center1
        aosym = 's1'
        partition_iorj = 'i'
        for j3c_order in ['Lij','ijL']:
            errs = test_gen(kmesh, scaled_center, aosym, j3c_order, partition_iorj)
            err = np.max(errs)
            self.assertAlmostEqual(err, 0., 10)


if __name__ == '__main__':
    print("Full Tests for rsdf_direct")
    unittest.main()
