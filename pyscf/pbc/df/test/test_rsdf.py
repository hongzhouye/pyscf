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
import numpy
from pyscf import lib
import pyscf.pbc
from pyscf import ao2mo, gto
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.df import rsdf

cell = pgto.Cell(
    atom="H 0 0 0; H 0.75 0 0",
    a = numpy.eye(3)*3,
    basis={"H": [[0,(0.5,1.)],[1,(0.3,1.)]]},
)
cell.verbose = 0
cell.max_memory = 1000
cell.build()
scaled_center = numpy.array([0.392, 0.105, 0.872])


def tearDownModule():
    global cell
    del cell

class KnownValues(unittest.TestCase):
    def test_h2_gamma(self):
        mydf = rsdf.RSDF(cell)
        mydf.build()
        mf = pscf.KRHF(cell).density_fit()
        mf.with_df = mydf
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -1.0430635251, 6)

    def test_h2_kpt1_shiftedcenter(self):
        kpts = cell.make_kpts([1,1,1], scaled_center=scaled_center)
        mydf = rsdf.RSDF(cell, kpts)
        mydf.build()
        mf = pscf.KRHF(cell, kpts).density_fit()
        mf.with_df = mydf
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -0.99618574676, 6)

    def test_h2_jonly_k211(self):
        kpts = cell.make_kpts([2,1,1])
        mydf = rsdf.RSDF(cell,kpts)
        mydf.build(j_only=True)
        mf = pscf.KRKS(cell,kpts).density_fit()
        mf.xc = "pbe"
        mf.with_df = mydf
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -1.00210235634, 6)

    def test_h2_jonly_k211_shiftedcenter(self):
        kpts = cell.make_kpts([2,1,1],scaled_center=scaled_center)
        mydf = rsdf.RSDF(cell,kpts)
        mydf.build(j_only=True)
        mf = pscf.KRKS(cell,kpts).density_fit()
        mf.xc = "pbe"
        mf.with_df = mydf
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -1.0047041613, 6)

    def test_h2_jk_k211(self):
        kpts = cell.make_kpts([2,1,1])
        mydf = rsdf.RSDF(cell,kpts)
        mydf.build()
        mf = pscf.KRHF(cell,kpts).density_fit()
        mf.with_df = mydf
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -0.98223442637, 6)

    def test_h2_jk_k211_shiftedcenter(self):
        kpts = cell.make_kpts([2,1,1],scaled_center=scaled_center)
        mydf = rsdf.RSDF(cell,kpts)
        mydf.build()
        mf = pscf.KRHF(cell,kpts).density_fit()
        mf.with_df = mydf
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -0.98409805858, 6)


if __name__ == '__main__':
    print("Full Tests for rsdf")
    unittest.main()
