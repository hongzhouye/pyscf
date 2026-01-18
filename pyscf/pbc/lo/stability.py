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
#         Genzhi Yang <genzyang17@gmail.com>
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

    tril_ijdx = numpy.tril_indices(mlo.norb, k=-1)
    tril_idx, tril_jdx = tril_ijdx
    thetas = numpy.asarray([1,3,5,7])*0.25*numpy.pi
    c2ts = numpy.cos(thetas*2)
    s2ts = numpy.sin(thetas*2)

    def update_rotation_local_(u, theta, i, j):
        for x in u:
            xi = x[:,i].copy()
            xj = x[:,j].copy()
            x[:,i] = xi*numpy.cos(theta) + xj*numpy.sin(theta)
            x[:,j] = -xi*numpy.sin(theta) + xj*numpy.cos(theta)

    u = mlo.identity_rotation()
    stable = True
    while True:
        # TODO: adjust this after changing atomic pops
        proj = mlo.atomic_pops(u)

        Lij = lib.einsum('ktxij->xij', proj.real)
        Lji = Lij.transpose(0,2,1)
        Lii = lib.einsum('xii->xi', Lij)
        Lijji = Lij + Lji
        Liijj = Lii[:,:,None] - Lii[:,None,:]

        Aij = (Lijji**2 - Liijj**2).sum(axis=0)[tril_ijdx]
        Bij = (Lijji * Liijj).sum(axis=0)[tril_ijdx]
        dLijt = Aij[:,None] * s2ts**2*0.5 - Bij[:,None] * s2ts*c2ts
        dLij = dLijt.max(axis=-1)

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

            theta = thetas[dLijt[idx].argmax(axis=-1)]
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
