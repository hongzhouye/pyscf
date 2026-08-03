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


class TDM(lib.StreamObject):
    '''Truncated density matrix builder for periodic HF exchange.

    Attributes:
        dm_trunc_level : str
            Level at which the density matrix is truncated. It can be
            ``'atom'`` or ``'cell'``. Default is ``'atom'``.
        dm_trunc_shape : str
            Shape of the truncation region. It can be ``'ws'`` or ``'sph'``.
            Default is ``'ws'``.
        ws_weight : bool
            Whether to weight density-matrix blocks on the Wigner--Seitz
            boundary. Default is False.
        direct_scf_tol : float
            Screening threshold for the exchange build. Default is
            ``cell.precision * 0.1``.
        dm_cond : str
            Density-matrix shell-block condition used for screening. It can
            be ``'absmax'``, ``'norm'``, or ``'abssum'``. Default is
            ``'norm'``.
        use_qqr : bool
            Whether to use distance-dependent QQR screening. If False, only
            QQ screening is used. Default is True.

    For conservative screening, set ``dm_cond = 'abssum'`` and
    ``use_qqr = False``.
    '''

    _keys = {
        'cell', 'kpts', 'dm_trunc_level', 'dm_trunc_shape', 'ws_weight',
        'dm_rcut', 'ws_search_mesh', 'direct_scf_tol', 'extent_tol',
        'dm_cond', 'use_qqr', 'profile',
    }

    def __init__(self, cell, kpts=np.zeros((1, 3))):
        self.cell = cell
        self.kpts = np.reshape(kpts, (-1, 3))
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory

        # Primary control attributes
        self.dm_trunc_level = 'atom'
        self.dm_trunc_shape = 'ws'
        self.ws_weight = False

        # Accuracy control attributes
        self.direct_scf_tol = cell.precision * .1
        self.dm_cond = 'norm'
        self.use_qqr = True

        # Do not set these attributes unless you know what you are doing
        self.extent_tol = .1
        self.dm_rcut = None
        self.ws_search_mesh = None
        self.profile = False

    def dump_flags(self, verbose=None):
        log = lib.logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info('dm_truncation_level = %s', self.dm_trunc_level)
        log.info('dm_truncation_shape = %s', self.dm_trunc_shape)
        if self.dm_trunc_shape == 'ws':
            log.info('ws_weight = %s', self.ws_weight)
        if self.dm_rcut is not None:
            log.info('dm_rcut = %g', self.dm_rcut)
        if self.ws_search_mesh is not None:
            log.info('ws_search_mesh = %g', self.ws_search_mesh)
        log.info('direct_scf_tol = %g', self.direct_scf_tol)
        log.info('dm_cond = %s', self.dm_cond)
        log.info('use_qqr = %s', self.use_qqr)
        if self.use_qqr:
            log.info('extent_tol = %g', self.extent_tol)
        if self.profile:
            log.info('TDM profiling = %s', self.profile)
        return self

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
            self.stdout = cell.stdout
            self.verbose = cell.verbose
            self.max_memory = cell.max_memory
        return self

    def get_k(self, dm, hermi=1, kpts=None, kpts_band=None, omega=None):
        from pyscf.pbc.tdm import tdm_jk
        return tdm_jk.get_k(
            self, dm, hermi, kpts, kpts_band, omega)
