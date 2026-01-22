#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
#         Gengzhi Yang <genzyang17@gmail.com>
#


import numpy

from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__

from pyscf.lo.stability import stability_newton


def stability_jacobi(mlo, verbose=None, return_status=False):
    ''' Check whether Jacobi sweep.
    '''
    log = logger.new_logger(mlo, verbose)
    exponent = mlo.exponent

    tril_ijdx = numpy.tril_indices(mlo.norb, k=-1)
    tril_idx, tril_jdx = tril_ijdx
    thetapool = numpy.asarray([1,2,3])*0.25*numpy.pi

    def update_rotation_local_(u, theta, i, j):
        for x in u:
            xi = x[:,i].copy()
            xj = x[:,j].copy()
            x[:,i] = xi*numpy.cos(theta) + xj*numpy.sin(theta)
            x[:,j] = -xi*numpy.sin(theta) + xj*numpy.cos(theta)

    u = mlo.identity_rotation()
    stable = True
    while True:
        Pij = mlo.atomic_pops(u, mode='00').real
        Qi = lib.einsum('xii->xi', Pij)
        Qiexp = Qi**exponent
        Lij = (Qiexp[:,None,:] + Qiexp[:,:,None]).sum(axis=0)[tril_ijdx]
        dLij = numpy.zeros_like(Lij)
        thetas = numpy.zeros_like(Lij)

        for theta in thetapool:
            c = numpy.cos(theta)
            s = numpy.sin(theta)

            Qitild = (Qi*c**2)[:,:,None] + (Qi*s**2)[:,None,:] + 2*c*s*Pij
            Qjtild = (Qi*s**2)[:,:,None] + (Qi*c**2)[:,None,:] - 2*c*s*Pij
            dLijtild = (Qitild**exponent+Qjtild**exponent).sum(axis=0)[tril_ijdx] - Lij
            mask = dLijtild > dLij + mlo.conv_tol
            thetas[mask] = theta
            dLij[mask] = dLijtild[mask]

        idxs = numpy.where(dLij > mlo.conv_tol)[0]

        if idxs.size == 0:
            break

        # Remove overlapping pairs using a greedy algorithm
        stable = False
        done = numpy.zeros(mlo.norb, dtype=bool)
        for idx in idxs:
            i, j = tril_idx[idx], tril_jdx[idx]
            if done[i] or done[j]:
                continue
            done[i] = done[j] = True

            theta = thetas[idx]
            log.info('Rotating orbital pair (%d,%d) by %.2f Pi. delta_f= %.14g',
                      i, j, theta/numpy.pi, dLij[idx])
            update_rotation_local_(u, theta, i, j)

    if stable:
        log.info(f'{mlo.__class__.__name__} is stable in the Jacobi stability analysis')
        mo_coeff = mlo.mo_coeff
    else:
        mo_coeff = mlo.rotate_orb(u)

    if return_status:
        return mo_coeff, stable
    else:
        return mo_coeff
