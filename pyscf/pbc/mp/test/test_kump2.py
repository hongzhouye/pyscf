#!/usr/bin/env python
# Copyright 2021 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy as np
from pyscf import __config__
from pyscf.pbc import gto, scf, mp


class Diamond_GDF(unittest.TestCase):
    ''' Diamond 311 kpt sampling with total spin = 2
        The number of occupied bands is [5,4,4] for spin up and [3,4,4] for spin down.
        The system can thus test both spin polarization and padding.
    '''
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.verbose = 6
        cell.output = '/dev/null'
        cell.atom = 'C 0 0 0; C 0.8925000000 0.8925000000 0.8925000000'
        cell.a = '''
        1.7850000000 1.7850000000 0.0000000000
        0.0000000000 1.7850000000 1.7850000000
        1.7850000000 0.0000000000 1.7850000000
        '''
        cell.pseudo = 'gth-hf-rev'
        cell.basis = {'C': [[0, (0.8, 1.0)], [0, (0.4, 1.0)],
                            [1, (1.0, 1.0)], [1, (0.5, 1.0)]]}
        cell.precision = 1e-12
        cell.spin = 2
        cell.build()
        kpts = cell.make_kpts((3,1,1))
        mf = scf.KUHF(cell, kpts=kpts).density_fit(auxbasis='weigend').run()
        cls.cell = cell
        cls.mf = mf
    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        del cls.cell, cls.mf

    def test_energy(self):
        mmp = mp.KMP2(self.mf).run()
        self.assertAlmostEqual(mmp.e_corr, -0.19856353453539324, 6)

    def test_energy_input_mo(self):
        # force recalculate mo energy from fock build
        mo_coeff = [c.copy() for c in self.mf.mo_coeff]
        mmp = mp.KMP2(self.mf).run(mo_coeff=mo_coeff)
        self.assertAlmostEqual(mmp.e_corr, -0.19856353453539324, 6)

    def test_energy_outcore(self):
        mmp = mp.KMP2(self.mf)
        mmp.max_memory = 2. # incore memory ~ 2.7 MB
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, -0.19856353453539324, 6)

    def test_energy_frozen(self):
        mmp = mp.KMP2(self.mf, frozen=1).run()
        self.assertAlmostEqual(mmp.e_corr, -0.1667849342139188, 6)

    def test_energy_C_kernel(self):
        mmp = mp.KMP2(self.mf)
        mmp._kernel = 'C'
        mmp.kernel(with_t2=False)
        self.assertAlmostEqual(mmp.e_corr, -0.19856353453539324, 6)


class Diamond_FFTDF(unittest.TestCase):
    ''' Diamond 311 kpt sampling with total spin = 2
        The number of occupied bands is [5,4,4] for spin up and [3,4,4] for spin down.
        The system can thus test both spin polarization and padding.
    '''
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.verbose = 6
        cell.output = '/dev/null'
        cell.atom = 'C 0 0 0; C 0.8925000000 0.8925000000 0.8925000000'
        cell.a = '''
        1.7850000000 1.7850000000 0.0000000000
        0.0000000000 1.7850000000 1.7850000000
        1.7850000000 0.0000000000 1.7850000000
        '''
        cell.pseudo = 'gth-hf-rev'
        cell.basis = {'C': [[0, (0.8, 1.0)], [0, (0.4, 1.0)],
                            [1, (1.0, 1.0)], [1, (0.5, 1.0)]]}
        cell.precision = 1e-10
        cell.spin = 2
        cell.build()
        cell.mesh = [21] * 3
        kpts = cell.make_kpts((3,1,1))
        mf = scf.KUHF(cell, kpts=kpts).run()
        cls.cell = cell
        cls.mf = mf
    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        del cls.cell, cls.mf

    def test_energy(self):
        mmp = mp.KMP2(self.mf).run()
        self.assertAlmostEqual(mmp.e_corr, -0.1985976917815226, 6)

    def test_energy_frozen(self):
        mmp = mp.KMP2(self.mf, frozen=1).run()
        self.assertAlmostEqual(mmp.e_corr, -0.16680211369767659, 6)


if __name__ == "__main__":
    print("Full Tests for KUMP2")
    unittest.main()
