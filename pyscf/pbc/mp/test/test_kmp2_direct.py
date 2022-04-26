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

import tempfile
import unittest
import numpy as np

from pyscf.pbc import gto, scf, mp
from pyscf import lib


atom = 'He 0 0 0; He 1 0 0'
a = np.eye(3) * 3
basis = 'cc-pvdz'
verbose = 6
cell = gto.Cell(atom=atom, a=a, basis=basis).set(verbose=verbose)
cell.build()

def run_scf(kmesh, direct=False, dm0=None):
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.with_df.direct = direct
    mf.kernel(dm0=dm0)
    return mf

def run_mp2(mf, frozen=None, with_t2=False):
    mmp = mp.KMP2(mf, frozen=frozen)
    mmp.kernel(with_t2=with_t2)
    return mmp


class KnownValues(unittest.TestCase):
    def test_211_energy(self):
        kmesh = (2,1,1)

        mf = run_scf(kmesh)
        mmp = run_mp2(mf)

        dm0 = mf.make_rdm1()
        mf_direct = run_scf(kmesh, direct=True, dm0=dm0)
        mmp_direct = run_mp2(mf_direct)

        self.assertAlmostEqual(mmp._scf.e_tot, mmp_direct._scf.e_tot, 8)
        self.assertAlmostEqual(mmp.e_corr, mmp_direct.e_corr, 8)

    def test_211_restart(self):
        kmesh = (2,1,1)
        frozen = None
        with_t2 = False

        mf = run_scf(kmesh)
        dm0 = mf.make_rdm1()
        mf_direct = run_scf(kmesh, direct=True, dm0=dm0)

        ftemp = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        mf_direct.max_memory = 0.05
        mmp1 = mp.KMP2(mf_direct, frozen=frozen)
        mmp1._cderi_to_save = ftemp.name
        mmp1.kernel(with_t2=with_t2)

        mmp2 = mp.KMP2(mf_direct, frozen=frozen)
        mmp2._cderi = ftemp.name
        mmp2.kernel(with_t2=with_t2)

        self.assertAlmostEqual(mmp1.e_corr, mmp2.e_corr, 8)

    def test_211_restart_ijL(self):
        kmesh = (2,1,1)
        frozen = None
        with_t2 = False

        mf = run_scf(kmesh)
        dm0 = mf.make_rdm1()
        mf_direct = run_scf(kmesh, direct=True, dm0=dm0)

        ftemp = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        mf_direct.max_memory = 0.05
        mmp1 = mp.KMP2(mf_direct, frozen=frozen)
        mmp1.j3c_order = 'ijL'
        mmp1._cderi_to_save = ftemp.name
        mmp1.kernel(with_t2=with_t2)

        mmp2 = mp.KMP2(mf_direct, frozen=frozen)
        mmp2.j3c_order = 'ijL'
        mmp2._cderi = ftemp.name
        mmp2.kernel(with_t2=with_t2)

        self.assertAlmostEqual(mmp1.e_corr, mmp2.e_corr, 8)

    def test_211_kilist(self):
        kmesh = (2,1,1)
        frozen = None
        with_t2 = False

        mf = run_scf(kmesh)
        dm0 = mf.make_rdm1()
        mf_direct = run_scf(kmesh, direct=True, dm0=dm0)
        mf_direct.verbose = 7

        nkpts = len(mf.kpts)
        mmp1 = mp.KMP2(mf_direct, frozen=frozen)
        mmp1.verbose = 7
        mmp1.kernel(with_t2=with_t2)

        mmp2 = mp.KMP2(mf_direct, frozen=frozen)
        mmp2.kilist = range(nkpts//2)
        mmp2.kernel(with_t2=with_t2)
        ecorr2 = mmp2.e_corr

        mmp2.kilist = range(nkpts//2,nkpts)
        mmp2.kernel(with_t2=with_t2)
        ecorr2 += mmp2.e_corr

        self.assertAlmostEqual(mmp1.e_corr, ecorr2, 8)


if __name__ == '__main__':
    print("Full KMP2 direct test")
    unittest.main()
