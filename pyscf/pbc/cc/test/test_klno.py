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
from pyscf.pbc import gto, scf, mp, cc
from pyscf import lo
from pyscf.cc.ccsd_t import kernel as CCSD_T
from pyscf.cc import LNOCCSD_T
from pyscf.pbc.cc import KLNOCCSD_T
from pyscf.cc.lno_helper import autofrag_iao
from pyscf.pbc.cc.klno_helper import k2s_scf, sort_orb_by_cell


class Water_In_A_Box(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.verbose = 4
        cell.output = '/dev/null'
        cell.atom = '''
        O   -1.485163346097   -0.114724564047    0.000000000000
        H   -1.868415346097    0.762298435953    0.000000000000
        H   -0.533833346097    0.040507435953    0.000000000000
        '''
        cell.a = np.eye(3) * 4
        cell.basis = 'cc-pvdz'
        cell.precision = 1e-10
        cell.build()

        kmesh = [3,1,1]
        kpts = cell.make_kpts(kmesh)
        nkpts = len(kpts)

        kmf = scf.KRHF(cell, kpts=kpts).density_fit().run()

        frozen_per_cell = 1
        frozen = frozen_per_cell * nkpts

        cls.cell = cell
        cls.kmf = kmf
        cls.frozen = frozen
    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        del cls.cell, cls.kmf, cls.frozen

    def test_lno_pm_by_thresh(self):
        cell = self.cell
        kmf = self.kmf
        frozen = self.frozen
        kpts = kmf.kpts

        mf = k2s_scf(kmf)

        # PM localization
        orbocc = mf.mo_coeff[:,frozen:np.count_nonzero(mf.mo_occ)]
        mlo = lo.PipekMezey(mf.cell, orbocc)
        lo_coeff = mlo.kernel()
        while True: # always performing jacobi sweep to avoid trapping in local minimum/saddle point
            lo_coeff1 = mlo.stability_jacobi()[1]
            if lo_coeff1 is lo_coeff:
                break
            mlo = lo.PipekMezey(mf.cell, lo_coeff1).set(verbose=4)
            mlo.init_guess = None
            lo_coeff = mlo.kernel()

        # Fragment list: for PM, every orbital corresponds to a fragment
        s1e = mf.get_ovlp()
        Nk = len(kpts)
        nlo = lo_coeff.shape[1]//Nk
        lo_coeff = sort_orb_by_cell(mf.cell, lo_coeff, Nk, s=s1e)
        frag_lolist = [[i] for i in range(nlo)]

        gamma = 10
        threshs = [1e-5,1e-6,1e-100]
        refs = [
            [-0.1998019819,-0.2102871047,-0.2132242357],
            [-0.2003627897,-0.2107978384,-0.2138505827],
            [-0.2005167756,-0.2109109734,-0.2140042176] # canonical
        ]
        for thresh,ref in zip(threshs,refs):
            mcc = KLNOCCSD_T(kmf, lo_coeff, frag_lolist, frozen=frozen, mf=mf).set(verbose=5)
            mcc.lno_thresh = [thresh*10,thresh]
            mcc.kernel()
            emp2 = mcc.e_corr_pt2
            eccsd = mcc.e_corr_ccsd
            eccsd_t = mcc.e_corr_ccsd_t
            # print('[%s],' % (','.join([f'{x:.10f}' for x in [emp2,eccsd,eccsd_t]])))
            self.assertAlmostEqual(emp2, ref[0], 6)
            self.assertAlmostEqual(eccsd, ref[1], 6)
            self.assertAlmostEqual(eccsd_t, ref[2], 6)

    def test_lno_iao_by_thresh(self):
        cell = self.cell
        kmf = self.kmf
        frozen = self.frozen
        kpts = kmf.kpts

        mf = k2s_scf(kmf)

        # IAO localization
        orbocc = mf.mo_coeff[:,frozen:np.count_nonzero(mf.mo_occ)]
        iao_coeff = lo.iao.iao(mf.cell, orbocc)
        lo_coeff = lo.orth.vec_lowdin(iao_coeff, mf.get_ovlp())
        celliao = lo.iao.reference_mol(mf.cell)

        # Fragment list: all IAOs belonging to same atom form a fragment
        frag_lolist_full = autofrag_iao(celliao)
        Nk = len(kmf.kpts)
        nfrag = len(frag_lolist_full)//Nk
        frag_lolist = frag_lolist_full[:Nk]

        gamma = 10
        threshs = [1e-5,1e-6,1e-100]
        refs = [
            [-0.2002220838,-0.2106579848,-0.2137039612],
            [-0.2004567566,-0.2108616407,-0.2139342959],
            [-0.2005167756,-0.2109109734,-0.2140042176] # canonical
        ]
        for thresh,ref in zip(threshs,refs):
            mcc = KLNOCCSD_T(kmf, lo_coeff, frag_lolist, frozen=frozen, mf=mf).set(verbose=5)
            mcc.lno_thresh = [thresh*10,thresh]
            mcc.kernel()
            emp2 = mcc.e_corr_pt2
            eccsd = mcc.e_corr_ccsd
            eccsd_t = mcc.e_corr_ccsd_t
            # print('[%s],' % (','.join([f'{x:.10f}' for x in [emp2,eccsd,eccsd_t]])))
            self.assertAlmostEqual(emp2, ref[0], 6)
            self.assertAlmostEqual(eccsd, ref[1], 6)
            self.assertAlmostEqual(eccsd_t, ref[2], 6)



if __name__ == "__main__":
    print("Full Tests for KLNO-CCSD and KLNO-CCSD(T)")
    unittest.main()
