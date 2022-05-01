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

''' Check if the direct RSDF JKbuild gives same results as the indirect one.
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


def get_dm(cell, kpts):
    from pyscf.pbc import scf
    from pyscf.pbc.lib.kpts_helper import is_zero
    nkpts = len(kpts)
    mf = scf.KRHF(cell, kpts)
    dm_kpts = mf.get_init_guess(key='minao')
    if not is_zero(kpts):   # non-Gamma point --> complex dm
        dm_kpts = dm_kpts + dm_kpts*0.1j
        for k in range(nkpts):
            e, u = scipy.linalg.eigh(dm_kpts[k])
            dm_kpts[k] = np.dot(u*abs(e), u.T.conj())
    return dm_kpts

def test_j(kmesh, scaled_center):
    nao = cell.nao_nr()
    kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)
    nkpts = len(kpts)

    dm_kpts = get_dm(cell, kpts)

    j_only = True

    mydf = df.RSDF(cell, kpts)
    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    mydf._cderi_to_save = swapfile.name
    swapfile = None
    mydf.build(j_only=j_only)

    vj_ref_kpts = mydf.get_jk(dm_kpts, kpts=kpts, with_j=True, with_k=False)[0]

    mydf2 = df.RSDF(cell, kpts)
    mydf2.direct = True
    mydf2.build()

    vj_kpts = mydf2.get_jk(dm_kpts, kpts=kpts, with_j=True, with_k=False)[0]

    same_shape = vj_ref_kpts.shape == vj_kpts.shape
    errs = [abs(vj_ref_kpts[k] - vj_kpts[k]).max() for k in range(nkpts)]

    return np.asarray(errs), same_shape

def test_k(kmesh, scaled_center, mydf2_kwargs={}):
    nao = cell.nao_nr()
    kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)
    nkpts = len(kpts)

    dm_kpts = get_dm(cell, kpts)

    j_only = False

    mydf = df.RSDF(cell, kpts)
    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    mydf._cderi_to_save = swapfile.name
    swapfile = None
    mydf.build(j_only=j_only)

    vk_ref_kpts = mydf.get_jk(dm_kpts, kpts=kpts, with_j=False, with_k=True)[1]

    mydf2 = df.RSDF(cell, kpts).set(**mydf2_kwargs)
    mydf2.direct = True
    mydf2.build()

    vk_kpts = mydf2.get_jk(dm_kpts, kpts=kpts, with_j=False, with_k=True)[1]

    same_shape = vk_ref_kpts.shape == vk_kpts.shape
    errs = [abs(vk_ref_kpts[k] - vk_kpts[k]).max() for k in range(nkpts)]

    return np.asarray(errs), same_shape


class KnownValues(unittest.TestCase):
    def test_j_kpt_unshifted(self):
        ''' Gamma
        '''
        kmesh = [1,1,1]
        scaled_center = scaled_center0
        errs, same_shape = test_j(kmesh, scaled_center)
        err = np.max(errs)
        self.assertAlmostEqual(err, 0., 10)
        self.assertTrue(same_shape)

    def test_j_kpt_shifted(self):
        ''' Single twisted angle
        '''
        kmesh = [1,1,1]
        scaled_center = scaled_center1
        errs, same_shape = test_j(kmesh, scaled_center)
        err = np.max(errs)
        self.assertAlmostEqual(err, 0., 10)
        self.assertTrue(same_shape)

    def test_j_kpts_unshifted(self):
        ''' Gamma-included kmesh
        '''
        kmesh = [3,2,1]
        scaled_center = scaled_center0
        errs, same_shape = test_j(kmesh, scaled_center)
        err = np.max(errs)
        self.assertAlmostEqual(err, 0., 10)
        self.assertTrue(same_shape)

    def test_j_kpts_shifted(self):
        ''' Twisted kmesh
        '''
        kmesh = [3,2,1]
        scaled_center = scaled_center1
        errs, same_shape = test_j(kmesh, scaled_center)
        err = np.max(errs)
        self.assertAlmostEqual(err, 0., 10)
        self.assertTrue(same_shape)

    def test_k_kpt_unshifted(self):
        ''' Gamma
        '''
        kmesh = [1,1,1]
        scaled_center = scaled_center0
        errs, same_shape = test_k(kmesh, scaled_center)
        err = np.max(errs)
        self.assertAlmostEqual(err, 0., 10)
        self.assertTrue(same_shape)

    def test_k_kpt_shifted(self):
        ''' Single twisted angle
        '''
        kmesh = [1,1,1]
        scaled_center = scaled_center1
        errs, same_shape = test_k(kmesh, scaled_center)
        err = np.max(errs)
        self.assertAlmostEqual(err, 0., 10)
        self.assertTrue(same_shape)

    def test_k_kpts_unshifted(self):
        ''' Gamma-included kmesh
        '''
        kmesh = [3,2,1]
        scaled_center = scaled_center0
        errs, same_shape = test_k(kmesh, scaled_center)
        err = np.max(errs)
        self.assertAlmostEqual(err, 0., 10)
        self.assertTrue(same_shape)

    def test_k_kpts_shifted(self):
        ''' Twisted kmesh
        '''
        kmesh = [3,2,1]
        scaled_center = scaled_center1
        errs, same_shape = test_k(kmesh, scaled_center)
        err = np.max(errs)
        self.assertAlmostEqual(err, 0., 10)
        self.assertTrue(same_shape)

    def test_k_kpt_unshifted_semidirect(self):
        ''' Gamma, semidirect
        '''
        kmesh = [1,1,1]
        scaled_center = scaled_center0
        errs, same_shape = test_k(kmesh, scaled_center, {'semidirect':True})
        err = np.max(errs)
        self.assertAlmostEqual(err, 0., 10)
        self.assertTrue(same_shape)

    def test_k_kpt_shifted_semidirect(self):
        ''' Single twisted angle, semidirect
        '''
        kmesh = [1,1,1]
        scaled_center = scaled_center1
        errs, same_shape = test_k(kmesh, scaled_center, {'semidirect':True})
        err = np.max(errs)
        self.assertAlmostEqual(err, 0., 10)
        self.assertTrue(same_shape)

    def test_k_kpts_unshifted_semidirect(self):
        ''' Gamma-included kmesh, semidirect
        '''
        kmesh = [3,2,1]
        scaled_center = scaled_center0
        errs, same_shape = test_k(kmesh, scaled_center, {'semidirect':True})
        err = np.max(errs)
        self.assertAlmostEqual(err, 0., 10)
        self.assertTrue(same_shape)

    def test_k_kpts_shifted_semidirect(self):
        ''' Twisted kmesh, semidirect
        '''
        kmesh = [3,2,1]
        scaled_center = scaled_center1
        errs, same_shape = test_k(kmesh, scaled_center, {'semidirect':True})
        err = np.max(errs)
        self.assertAlmostEqual(err, 0., 10)
        self.assertTrue(same_shape)


if __name__ == '__main__':
    print("Full Tests for rsdf_direct")
    unittest.main()
