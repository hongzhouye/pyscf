#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#

import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__

from pyscf.soscf.ciah import CIAHOptimizerMixin, CIAHOptimizerMixinComplex, expmat


class SubspaceCIAHOptimizerMixin(CIAHOptimizerMixin):
    def __init__(self, norb, rtypes):
        self.norb = norb
        self.rtypes = rtypes

    @property
    def pdim(self):
        n = self.norb
        pdim_map = {0: n*(n-1)//2, 1: n*(n-1), 2: n*n}
        return sum([pdim_map[x] for x in self.rtypes])

    @property
    def sdim(self):
        return len(self.rtypes)

    def pack_uniq_var(self, mat):
        mat = numpy.reshape(mat, (self.sdim, self.norb, self.norb))
        tril_idx = numpy.tril_indices(self.norb, k=-1)
        v = []
        for i in range(self.sdim):
            v.append( mat[i].real[tril_idx] )
            if self.rtypes[i] > 0:
                v.append( mat[i].imag[tril_idx] )
            if self.rtypes[i] > 1:
                v.append( numpy.diag(mat[i].imag) )
        return numpy.hstack(v)

    def unpack_uniq_var(self, v):
        v = numpy.asarray(v).reshape(-1)
        assert( v.size == self.pdim )
        n = self.norb
        n2 = n*(n-1)//2
        tril_idx = numpy.tril_indices(n, k=-1)
        mat = numpy.zeros((self.sdim, n,n), dtype=numpy.complex128) # real if all type 0?
        p1 = 0
        for i in range(self.sdim):
            M = mat[i]
            p0, p1 = p1, p1+n2
            M[tril_idx] = v[p0:p1]
            if self.rtypes[i] > 0:
                p0, p1 = p1, p1+n2
                M[tril_idx] += v[p0:p1] * 1j
            if self.rtypes[i] > 1:
                p0, p1 = p1, p1+n
                numpy.fill_diagonal(M, v[p0:p1] * 0.5j)
            M -= M.T.conj()
        return mat

    def extract_rotation(self, dr, u0=None):
        dr = self.unpack_uniq_var(dr)
        u1 = numpy.asarray([expmat(drk) for drk in dr])
        if u0 is None:
            return u1
        else:
            return self.update_rotation(u0, u1)

    def update_rotation(self, u0, u1):
        return numpy.asarray([numpy.dot(u0k, u1k) for u0k,u1k in zip(u0,u1)])

    def zero_uniq_var(self):
        return numpy.zeros(self.pdim)


class KptsCIAHOptimizerMixin(CIAHOptimizerMixin):
    '''
    Real, translationally symmetric orbital rotations in a supercell,
    parameterized using k-point–resolved generators.

    For the supercell rotation to be real and translationally symmetric,
    the k-point set must be closed under inversion, i.e.,
        for all k in K,  -k (mod G) is also in K,
    where G denotes a reciprocal lattice vector.

    Under this condition, the Nk k-points can be partitioned into two subsets:
      1. Time-reversal invariant (TRI) k-points, satisfying k = -k (mod G).
      2. Non-TRI k-points, which occur in inversion-related pairs (k, k'),
         with k = -k' (mod G).

    The generator matrices have the following structure:
      * For TRI k-points, the generators are real and antisymmetric, each
        parameterized by n*(n-1)/2 real parameters.
      * For non-TRI k-points, the generators are complex and anti-Hermitian,
        each parameterized by n^2 real parameters. Only one k-point from each
        inversion-related pair needs to be explicitly parameterized, because
            K[k'] = K[k].conj()  if k' = -k (mod G)

    Let Ns denote the number of TRI k-points. The total number of real
    parameters is then
        Ns * n*(n-1)/2 + (Nk - Ns)/2 * n^2 = (Nk*n^2 - Ns*n) / 2.

    Args:
        norb (int):
            Number of bands (orbitals) to be mixed per k-point.
        kpairs (list):
            A list classifying k-points into self-inversion singletons and
            inversion-related pairs. For example,
                [[0], [1, 3], [2, 4], [5], [6], [7]]
            indicates that k-points 0, 5, 6, and 7 are self-inversion (TRI),
            while (1, 3) and (2, 4) form inversion-related pairs.
    '''

    def __init__(self, kpts, norb, kpairs=None):
        self.kpts = kpts
        self.norb = norb
        self.kpairs = kpairs
        if kpairs is not None:
            assert( sum([len(x) for x in kpairs]) == self.kdim )

    @property
    def kdim(self):
        return len(self.kpts)

    @property
    def pdim(self):
        n = self.norb
        if self.kpairs is None:
            return self.kdim*n**2 - n
        else:
            nself = sum([len(kpair)==1 for kpair in self.kpairs])
            npair = len(self.kpairs) - nself
            return nself*n*(n-1)//2 + npair*n**2

    def pack_uniq_var(self, mat):
        mat = numpy.reshape(mat, (self.kdim, self.norb, self.norb))
        tril_idx = numpy.tril_indices(self.norb, k=-1)
        v = []
        if self.kpairs is None:
            for k in range(self.kdim):
                v.append( mat[k].real[tril_idx] )
                v.append( mat[k].imag[tril_idx] )
                if k != 0:
                    v.append( numpy.diag(mat[k].imag) )
        else:
            for kpair in self.kpairs:
                k = kpair[0]
                v.append( mat[k].real[tril_idx] )
                if len(kpair) == 2:
                    v.append( mat[k].imag[tril_idx] )
                    v.append( numpy.diag(mat[k].imag) )
        return numpy.hstack(v)

    def unpack_uniq_var(self, v):
        v = v.reshape(-1)
        n = self.norb
        n2 = n*(n-1)//2
        tril_idx = numpy.tril_indices(n, k=-1)
        mat = numpy.zeros((self.kdim, n,n), dtype=numpy.complex128)
        if self.kpairs is None:
            p1 = 0
            for k in range(self.kdim):
                M = mat[k]
                p0, p1 = p1, p1+n2
                M[tril_idx] = v[p0:p1]
                p0, p1 = p1, p1+n2
                M[tril_idx] += v[p0:p1] * 1j
                if k != 0:
                    p0, p1 = p1, p1+n
                    numpy.fill_diagonal(M, v[p0:p1] * 0.5j)
                M -= M.T.conj()
        else:
            p1 = 0
            for kpair in self.kpairs:
                M = mat[kpair[0]]
                p0, p1 = p1, p1+n2
                M[tril_idx] = v[p0:p1]
                if len(kpair) == 2:
                    p0, p1 = p1, p1+n2
                    M[tril_idx] += v[p0:p1] * 1j
                    p0, p1 = p1, p1+n
                    numpy.fill_diagonal(M, v[p0:p1] * 0.5j)
                M -= M.T.conj()
                if len(kpair) == 2:
                    mat[kpair[1]] = M.conj()
        return mat

    def extract_rotation(self, dr, u0=None):
        dr = self.unpack_uniq_var(dr)
        u1 = numpy.asarray([expmat(drk) for drk in dr])
        if u0 is None:
            return u1
        else:
            return self.update_rotation(u0, u1)

    def update_rotation(self, u0, u1):
        return numpy.asarray([numpy.dot(u0k, u1k) for u0k,u1k in zip(u0,u1)])

    def zero_uniq_var(self):
        return numpy.zeros(self.pdim)


def KptsCIAHOptimizerMixinReal(KptsCIAHOptimizerMixin):
    def __init__(self, kpts, norb, kpairs):
        self.kpts = kpts
        self.norb = norb
        self.kpairs = kpairs
        assert( sum([len(x) for x in kpairs]) == self.kdim )

    @property
    def pdim(self):
        n = self.norb
        nself = sum([len(kpair)==1 for kpair in self.kpairs])
        npair = len(self.kpairs) - nself
        return nself*n*(n-1)//2 + npair*n**2

    def pack_uniq_var(self, mat):
        mat = numpy.reshape(mat, (self.kdim, self.norb, self.norb))
        tril_idx = numpy.tril_indices(self.norb, k=-1)
        v = []
        for kpair in self.kpairs:
            k = kpair[0]
            v.append( mat[k].real[tril_idx] )
            if len(kpair) == 2:
                v.append( mat[k].imag[tril_idx] )
                v.append( numpy.diag(mat[k].imag) )
        return numpy.hstack(v)

    def unpack_uniq_var(self, v):
        v = v.reshape(-1)
        n = self.norb
        n2 = n*(n-1)//2
        tril_idx = numpy.tril_indices(n, k=-1)
        mat = numpy.zeros((self.kdim, n,n), dtype=numpy.complex128)
        p1 = 0
        for kpair in self.kpairs:
            M = mat[kpair[0]]
            p0, p1 = p1, p1+n2
            M[tril_idx] = v[p0:p1]
            if len(kpair) == 2:
                p0, p1 = p1, p1+n2
                M[tril_idx] += v[p0:p1] * 1j
                p0, p1 = p1, p1+n
                numpy.fill_diagonal(M, v[p0:p1] * 0.5j)
            M -= M.T.conj()
            if len(kpair) == 2:
                mat[kpair[1]] = M.conj()
        return mat
