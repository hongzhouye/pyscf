#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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


import numpy as np

from pyscf import lib
from pyscf.pbc.lo.base import get_kmesh
from pyscf.pbc.lo.interpolation import get_WigerSeitz_Rs


def get_dm_Ls(cell, kpts, level='atom', shape='sph', ws_weight=True,
              rcut=None, search_mesh=None):
    if level not in ('cell', 'atom'):
        raise NotImplementedError('TDM truncation level %s' % level)

    if shape == 'sph':
        return get_dm_Ls_sph(cell, len(kpts), level, rcut)
    elif shape == 'ws':
        kmesh = get_kmesh(cell, kpts)
        return get_dm_Ls_ws(
            cell, kmesh, level, ws_weight, search_mesh)
    else:
        raise NotImplementedError('TDM truncation shape %s' % shape)


def get_dm_Ls_sph(cell, nkpts, level='atom', rcut=None):
    if rcut is None:
        rcut = (3*nkpts*cell.vol/(4*np.pi))**(1./3)
    nimgs = get_nimgs(cell, rcut)
    Ts = lib.cartesian_prod([np.arange(-x, x+1) for x in nimgs])
    Ls = np.dot(Ts, cell.lattice_vectors())

    natm = cell.natm
    if level == 'cell':
        weights = lib.norm(Ls, axis=1) < rcut
        atmweights = np.broadcast_to(
            weights[:,None,None], (len(Ls), natm, natm)).copy()
    elif level == 'atom':
        atmweights = _get_atmdist(cell, Ls) < rcut
    else:
        raise NotImplementedError

    idx = np.any(atmweights, axis=(1, 2))
    return Ls[idx], np.asarray(atmweights[idx], dtype=float)


def get_ws_weights(cell, kmesh, rs, search_mesh=None, tol=1e-6):
    '''Weights for vectors on the WS boundary of a BvK supercell.'''
    if search_mesh is None:
        search_mesh = [3, 3, 3]
    search_mesh = np.asarray(search_mesh)
    kmesh = np.asarray(kmesh)

    bvk_a = np.einsum('x,xy->xy', kmesh, cell.lattice_vectors())
    Ts = lib.cartesian_prod([
        np.arange(-x, x+1) for x in search_mesh])
    bvk_Ls = np.dot(Ts, bvk_a)
    i0 = np.argmin(lib.norm(bvk_Ls, axis=1))

    dist = lib.norm(np.asarray(rs)[:,None] - bvk_Ls, axis=2)
    dist_min = dist.min(axis=1)
    in_ws = abs(dist[:,i0] - dist_min) < tol
    ndegen = np.count_nonzero(
        abs(dist - dist_min[:,None]) < tol, axis=1)

    weights = np.zeros(len(rs))
    weights[in_ws] = 1./ndegen[in_ws]
    return weights


def get_dm_Ls_ws(cell, kmesh, level='atom', ws_weight=True,
                 search_mesh=None, tol=1e-6):
    if level not in ('cell', 'atom'):
        raise NotImplementedError('TDM truncation level %s' % level)
    if search_mesh is None:
        search_mesh = [3, 3, 3]
    search_mesh = np.asarray(search_mesh)
    kmesh = np.asarray(kmesh)
    nkpts = np.prod(kmesh)

    if level == 'cell':
        Ls = get_WigerSeitz_Rs(
            cell.lattice_vectors(), kmesh,
            search_mesh=search_mesh, tol=tol)[0]
        weights = get_ws_weights(
            cell, kmesh, Ls, search_mesh, tol)
        if abs(weights.sum() - nkpts) > tol:
            raise RuntimeError('WS weights do not sum to kmesh size.')
        if not ws_weight:
            weights[:] = 1
        atmweights = np.broadcast_to(
            weights[:,None,None], (len(Ls), cell.natm, cell.natm)).copy()
        return Ls, atmweights

    nimgs = search_mesh * kmesh
    Ts = lib.cartesian_prod([
        np.arange(-x, x+1) for x in nimgs])
    Ls = np.dot(Ts, cell.lattice_vectors())

    atom_coords = cell.atom_coords()
    natm = cell.natm
    atmweights = np.zeros((len(Ls), natm, natm))
    for ia in range(natm):
        for ja in range(natm):
            dr = Ls + atom_coords[ja] - atom_coords[ia]
            weights = get_ws_weights(
                cell, kmesh, dr, search_mesh, tol)
            if abs(weights.sum() - nkpts) > tol:
                raise RuntimeError(
                    'WS weights do not sum to kmesh size. Please use a '
                    'larger search_mesh than %s.' % list(search_mesh))
            atmweights[:,ia,ja] = weights

    if not ws_weight:
        atmweights[atmweights != 0] = 1
    idx = np.any(atmweights, axis=(1, 2))
    return Ls[idx], atmweights[idx]


def apply_weights(cell, atmweights, dm_Ls):
    nao = cell.nao_nr()
    aoslices = cell.aoslice_by_atom()[:,-2:]
    dm_weights = np.zeros((nao, nao))
    out = np.empty_like(dm_Ls)

    for iL, weights in enumerate(atmweights):
        dm_weights[:] = 0
        for ia in range(cell.natm):
            i0, i1 = aoslices[ia]
            for ja in np.where(weights[ia] != 0)[0]:
                j0, j1 = aoslices[ja]
                dm_weights[i0:i1,j0:j1] = weights[ia,ja]
        out[iL] = dm_weights * dm_Ls[iL]
    return out


def get_nimgs(cell, rcut):
    b = cell.reciprocal_vectors(norm_to=1)
    heights_inv = lib.norm(b, axis=1)
    scaled_coords = cell.atom_coords().dot(b.T)
    boundary = np.max([abs(scaled_coords).max(axis=0),
                       abs(1-scaled_coords).max(axis=0)], axis=0)
    nimgs = np.ceil(rcut*heights_inv + boundary).astype(int)

    if cell.dimension == 0:
        nimgs = [0, 0, 0]
    elif cell.dimension == 1:
        nimgs = [nimgs[0], 0, 0]
    elif cell.dimension == 2:
        nimgs = [nimgs[0], nimgs[1], 0]
    return np.asarray(nimgs)


def _get_atmdist(cell, Ls):
    atom_coords = cell.atom_coords()
    return lib.norm(Ls[:,None,None] + atom_coords[None,None]
                    - atom_coords[None,:,None], axis=-1)
