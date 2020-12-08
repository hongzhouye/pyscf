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

import ctypes
import numpy
from pyscf import lib

libpbc = lib.load_library('libpbc')
def _fpointer(name):
    return ctypes.addressof(getattr(libpbc, name))

class PBCOpt(object):
    def __init__(self, cell):
        self._this = ctypes.POINTER(_CPBCOpt)()
        natm = ctypes.c_int(cell._atm.shape[0])
        nbas = ctypes.c_int(cell._bas.shape[0])
        libpbc.PBCinit_optimizer(ctypes.byref(self._this),
                                 cell._atm.ctypes.data_as(ctypes.c_void_p), natm,
                                 cell._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                                 cell._env.ctypes.data_as(ctypes.c_void_p))

    def init_rcut_cond(self, cell, precision=None):
        if precision is None: precision = cell.precision
        rcut = numpy.array([cell.bas_rcut(ib, precision)
                            for ib in range(cell.nbas)])
        natm = ctypes.c_int(cell._atm.shape[0])
        nbas = ctypes.c_int(cell._bas.shape[0])
        libpbc.PBCset_rcut_cond(self._this,
                                rcut.ctypes.data_as(ctypes.c_void_p),
                                cell._atm.ctypes.data_as(ctypes.c_void_p), natm,
                                cell._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                                cell._env.ctypes.data_as(ctypes.c_void_p))
        return self

    def del_rcut_cond(self):
        self._this.contents.fprescreen = _fpointer('PBCnoscreen')
        return self

    def __del__(self):
        try:
            libpbc.PBCdel_optimizer(ctypes.byref(self._this))
        except AttributeError:
            pass

class _CPBCOpt(ctypes.Structure):
    _fields_ = [('rrcut', ctypes.c_void_p),
                ('rc_cut', ctypes.c_void_p),
                ('r12_cut', ctypes.c_void_p),
                ('bas_exp', ctypes.c_void_p),
                ('fprescreen', ctypes.c_void_p)]

# Hongzhou: testing a few prescreening conditions
class PBCOpt1(PBCOpt):  # R12Rc_max
    def __init__(self, cell):
        self._this = ctypes.POINTER(_CPBCOpt)()
        natm = ctypes.c_int(cell._atm.shape[0])
        nbas = ctypes.c_int(cell._bas.shape[0])
        libpbc.PBCinit_optimizer1(ctypes.byref(self._this),
                                  cell._atm.ctypes.data_as(ctypes.c_void_p), natm,
                                  cell._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                                  cell._env.ctypes.data_as(ctypes.c_void_p))

    def init_rcut_cond(self, cell, prescreening_data, precision=None):
        if precision is None: precision = cell.precision
        rcut = numpy.array([cell.bas_rcut(ib, precision)
                            for ib in range(cell.nbas)])
        natm = ctypes.c_int(cell._atm.shape[0])
        nbas = ctypes.c_int(cell._bas.shape[0])
        Rc_cut, R12_cut_lst = prescreening_data
        bas_exp = numpy.asarray([numpy.min(cell.bas_exp(ib))
                                for ib in range(cell.nbas)])
        libpbc.PBCset_rcut_cond1(self._this,
                                 ctypes.c_double(Rc_cut),
                                 R12_cut_lst.ctypes.data_as(ctypes.c_void_p),
                                 bas_exp.ctypes.data_as(ctypes.c_void_p),
                                 cell._atm.ctypes.data_as(ctypes.c_void_p), natm,
                                 cell._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                                 cell._env.ctypes.data_as(ctypes.c_void_p))
        return self

    def __del__(self):
        try:
            libpbc.PBCdel_optimizer1(ctypes.byref(self._this))
        except AttributeError:
            pass

class PBCOpt2(PBCOpt1):  # R12Rc_max
    def __init__(self, cell):
        PBCOpt1.__init__(self, cell)

    def init_rcut_cond(self, cell, prescreening_data, precision=None):
        if precision is None: precision = cell.precision
        rcut = numpy.array([cell.bas_rcut(ib, precision)
                            for ib in range(cell.nbas)])
        natm = ctypes.c_int(cell._atm.shape[0])
        nbas = ctypes.c_int(cell._bas.shape[0])
        Rc_cut_mat, R12_cut_mat = prescreening_data
        Rc_cut_mat = numpy.asarray(Rc_cut_mat, order="C")
        R12_cut_mat = numpy.asarray(R12_cut_mat, order="C")
        nbas_auxchg = ctypes.c_int(Rc_cut_mat.shape[0])
        nc_max = ctypes.c_int(R12_cut_mat.shape[-1])
        bas_exp = numpy.asarray([numpy.min(cell.bas_exp(ib))
                                for ib in range(cell.nbas)])
        libpbc.PBCset_rcut_cond2(self._this,
                                 nbas_auxchg, nc_max,
                                 Rc_cut_mat.ctypes.data_as(ctypes.c_void_p),
                                 R12_cut_mat.ctypes.data_as(ctypes.c_void_p),
                                 bas_exp.ctypes.data_as(ctypes.c_void_p),
                                 cell._atm.ctypes.data_as(ctypes.c_void_p), natm,
                                 cell._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                                 cell._env.ctypes.data_as(ctypes.c_void_p))
        return self
