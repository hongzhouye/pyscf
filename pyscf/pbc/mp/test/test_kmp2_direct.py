#!/usr/bin/env python
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

import unittest
import numpy as np

from pyscf.pbc import gto, scf, mp


atom = 'He 0 0 0; He 1 0 0'
a = np.eye(3) * 3
basis = 'cc-pvdz'
verbose = 0
cell = gto.Cell(atom=atom, a=a, basis=basis).set(verbose=verbose)
cell.build()

def run1(kmesh, direct=False, dm0=None, frozen=None, with_t2=False):
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.with_df.direct = direct
    mf.kernel(dm0=dm0)

    mmp = mp.KMP2(mf, frozen=frozen)
    mmp.kernel(with_t2=with_t2)

    return mmp



class KnownValues(unittest.TestCase):
    def test_211_energy(self):
        kmesh = (2,1,1)
        mmp = run1(kmesh)

        dm0 = mmp._scf.make_rdm1()
        mmp_direct = run1(kmesh, direct=True, dm0=dm0)

        self.assertAlmostEqual(mmp._scf.e_tot, mmp_direct._scf.e_tot, 8)
        self.assertAlmostEqual(mmp.e_corr, mmp_direct.e_corr, 8)


if __name__ == '__main__':
    print("Full KMP2 direct test")
    unittest.main()
