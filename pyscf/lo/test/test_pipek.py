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
from pyscf import gto, lo


class Water(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 4
        mol.output = '/dev/null'
        mol.atom = '''
        O     0.000000     0.000000     0.000000
        O     0.000000     0.000000     1.480000
        H     0.895669     0.000000    -0.316667
        H    -0.895669     0.000000     1.796667
        '''
        mol.basis = 'sto-3g'
        mol.precision = 1e-10
        mol.build()

        cls.mol = mol

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()
        del cls.mol

    def test_cost(self):
        ''' Test for `cost_function`
        '''
        def test1(mlo, loss_ref):
            for exponent in [2,3,4]:
                mlo.set(exponent=exponent)
                loss = mlo.cost_function()
                self.assertAlmostEqual(loss, loss_ref[exponent], 6)

        mol = self.mol

        # real orbitals
        s = mol.intor_symmetric('int1e_ovlp')
        mo_coeff = lo.orth.schmidt(s)
        norb = mo_coeff.shape[1]

        mlo = lo.pipek.PipekMezey(mol, mo_coeff)

        mlo.pop_method = 'meta-lowdin'
        loss_ref = {
            2: 11.0347269392,
            3: 10.5595998735,
            4: 10.1484720537,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'iao'
        loss_ref = {
            2: 11.1065896746,
            3: 10.6685385701,
            4: 10.2886276646,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'iao-biorth'
        loss_ref = {
            2: 12.4952121786,
            3: 12.7449891556,
            4: 13.0116401223,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'becke'
        loss_ref = {
            2: 9.4230597975,
            3: 8.2191241100,
            4: 7.3049902834,
        }
        test1(mlo, loss_ref)


        # complex orbitals
        mo_coeff = mo_coeff + np.cos(mo_coeff)*0.01j

        mlo = lo.pipek.PipekMezey(mol, mo_coeff)

        mlo.pop_method = 'meta-lowdin'
        loss_ref = {
            2: 11.0457506419,
            3: 10.5751208425,
            4: 10.1685286213,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'iao'
        loss_ref = {
            2: 11.1150294738,
            3: 10.6804270421,
            4: 10.3041148130,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'iao-biorth'
        loss_ref = {
            2: 12.5189728118,
            3: 12.7814008434,
            4: 13.0613620163,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'becke'
        loss_ref = {
            2: 9.4333937604,
            3: 8.2317523225,
            4: 7.3198256336,
        }
        test1(mlo, loss_ref)


    def test_grad_hess(self):
        ''' Test for `get_grad` and `gen_g_hop`
        '''
        mol = self.mol

        def test1(mo_coeff):
            norb = mo_coeff.shape[1]
            u0 = np.eye(norb)

            mlo = lo.pipek.PipekMezey(mol, mo_coeff)
            mlo.pop_method = 'meta-lowdin'

            x0 = mlo.zero_uniq_var()

            step_length = 1e-3
            precision = 4

            for exponent in [2,3,4]:
                mlo.set(exponent=exponent)
                g = mlo.get_grad(u0)
                g1, h_op, h_diag = mlo.gen_g_hop(u0)

                self.assertAlmostEqual(abs(g-g1).max(), 0, 6)

                H = np.zeros((x0.size,x0.size))
                for i in range(x0.size):
                    x0[i] = 1
                    H[:,i] = h_op(x0)
                    x0[i] = 0

                self.assertAlmostEqual(abs(np.diagonal(H)-h_diag).max(), 0, 6)

                def func(x):
                    u = mlo.extract_rotation(x)
                    return -mlo.cost_function(u)

                num_g = _num_grad(func, x0, step_length)
                num_H = _num_hess(func, x0, step_length)

                self.assertAlmostEqual(abs(g-num_g).max(), 0, precision)
                self.assertAlmostEqual(abs(H-num_H).max(), 0, precision)

        s = mol.intor_symmetric('int1e_ovlp')
        mo_coeff = lo.orth.schmidt(s)

        mo_idx = [1,6,9]
        mo_coeff = mo_coeff[:,mo_idx]
        test1(mo_coeff) # real orbitals
        test1(mo_coeff + np.cos(mo_coeff)*0.01j)    # complex orbitals


def test_cost(mol, pop_method, exponent, loss_ref):
    # real orbitals
    s = mol.intor_symmetric('int1e_ovlp')
    mo_coeff = lo.orth.schmidt(s)
    norb = mo_coeff.shape[1]

    mlo = lo.pipek.PipekMezey(mol, mo_coeff)
    mlo.exponent = exponent
    mlo.pop_method = pop_method
    loss = mlo.cost_function()

    return loss - loss_ref


def _num_grad(func, x0, step_length):
    x0 = np.asarray(x0)
    n = x0.size
    g = np.zeros_like(x0)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = step_length
        yf = func(x0+dx)
        yb = func(x0-dx)
        g[i] = (yf-yb) / (2.*step_length)
    return g
def _num_hess(func, x0, step_length):
    x0 = np.asarray(x0)
    n = x0.size
    H = np.zeros((n,n))
    y0 = func(x0)
    for i in range(n):
        dxi = np.zeros(n)
        dxi[i] = step_length
        for j in range(i+1,n):
            dxj = np.zeros(n)
            dxj[j] = step_length
            yff = func(x0+dxi+dxj)
            yfb = func(x0+dxi-dxj)
            ybf = func(x0-dxi+dxj)
            ybb = func(x0-dxi-dxj)
            H[i,j] = H[j,i] = (yff+ybb-yfb-ybf) / (4.*step_length**2.)
        yf = func(x0+dxi)
        yb = func(x0-dxi)
        H[i,i] = (yf+yb-2*y0) / step_length**2.

    return H


if __name__ == "__main__":
    print("Full Tests for PipekMezey")
    unittest.main()
