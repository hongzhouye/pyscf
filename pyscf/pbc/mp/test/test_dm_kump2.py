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

import unittest
from functools import reduce
import numpy as np
from pyscf.pbc import gto, scf, mp

def setUpModule():
    global cell, kmf, kpts, nkpts
    cell = gto.Cell()
    cell.atom = """
    O          0.00000        0.00000        0.11779
    H          0.00000        0.75545       -0.47116
    H          0.00000       -0.75545       -0.47116
    """
    cell.basis = 'cc-pvdz'
    cell.a = np.eye(3) * 4
    cell.spin = 4
    cell.output = '/dev/null'
    cell.build()

    nk = [2,1,1]
    kpts = cell.make_kpts(nk)
    nkpts = len(kpts)
    kmf = scf.KUHF(cell, kpts=kpts, exxdiv=None).density_fit()
    kmf.verbose = 4
    kmf.kernel()

def tearDownModule():
    global cell, kmf
    cell.stdout.close()
    del cell, kmf

class KnownValues(unittest.TestCase):
    def test_kump2_contract_eri_dm(self):
        kmp2 = mp.KMP2(kmf)
        kmp2.kernel()
        e_tot = kmp2.e_tot

        mo_coeff = kmf.mo_coeff
        hcore_ao = kmf.get_hcore()

        hcore = [[None]*nkpts, [None]*nkpts]
        for s in [0,1]:
            for k in range(nkpts):
                hcore[s][k] = reduce(np.dot, (mo_coeff[s][k].T.conj(), hcore_ao[k], mo_coeff[s][k]))

        dm1 = kmp2.make_rdm1()
        dm2 = kmp2.make_rdm2()
        e1 = 0
        for s in [0,1]:
            for k in range(nkpts):
                e1 += np.einsum('pq,qp', dm1[s][k], hcore[s][k]).real / nkpts
        ao2mo = kmp2._scf.with_df.ao2mo
        e2 = 0
        idx = 0
        for kp in range(nkpts):
            for kq in range(nkpts):
                for kr in range(nkpts):
                    ks = kmp2.khelper.kconserv[kp,kq,kr]

                    s1 = s2 = 0
                    dm2_ = dm2[0]
                    mop = mo_coeff[s1][kp]
                    moq = mo_coeff[s1][kq]
                    mor = mo_coeff[s2][kr]
                    mos = mo_coeff[s2][ks]
                    eri = ao2mo((mop,moq,mor,mos),
                          (kpts[kp], kpts[kq], kpts[kr], kpts[ks]),
                          compact=False).reshape(mop.shape[-1],moq.shape[-1],
                          mor.shape[-1],mos.shape[-1]) / nkpts
                    e2 += np.einsum('pqrs,pqrs',dm2_[idx], eri).real * 0.5 / nkpts

                    s1 = s2 = 1
                    dm2_ = dm2[2]
                    mop = mo_coeff[s1][kp]
                    moq = mo_coeff[s1][kq]
                    mor = mo_coeff[s2][kr]
                    mos = mo_coeff[s2][ks]
                    eri = ao2mo((mop,moq,mor,mos),
                          (kpts[kp], kpts[kq], kpts[kr], kpts[ks]),
                          compact=False).reshape(mop.shape[-1],moq.shape[-1],
                          mor.shape[-1],mos.shape[-1]) / nkpts
                    e2 += np.einsum('pqrs,pqrs',dm2_[idx], eri).real * 0.5 / nkpts

                    s1, s2 = 0, 1
                    dm2_ = dm2[1]
                    mop = mo_coeff[s1][kp]
                    moq = mo_coeff[s1][kq]
                    mor = mo_coeff[s2][kr]
                    mos = mo_coeff[s2][ks]
                    eri = ao2mo((mop,moq,mor,mos),
                          (kpts[kp], kpts[kq], kpts[kr], kpts[ks]),
                          compact=False).reshape(mop.shape[-1],moq.shape[-1],
                          mor.shape[-1],mos.shape[-1]) / nkpts
                    e2 += np.einsum('pqrs,pqrs',dm2_[idx], eri).real / nkpts

                    idx += 1
        e = e1 + e2 + cell.energy_nuc()
        self.assertAlmostEqual(e, e_tot, 4)

if __name__ == "__main__":
    print("Full Tests for kump2 rdm")
    unittest.main()
