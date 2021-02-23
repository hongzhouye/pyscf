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
#
# Author: Hong-Zhou Ye <osirpt.sun@gmail.com>
#

import h5py
from pyscf.lib.chkfile import load, save, save_mol
from pyscf.pbc.lib.chkfile import load_cell


def load_scf(chkfile):
    return load_cell(chkfile), load(chkfile, 'scf')


def dump_scf(mol, chkfile, e_tot, mo_energy, mo_occ, mo_coeff,
             overwrite_mol=True):
    if h5py.is_hdf5(chkfile) and not overwrite_mol:
        with h5py.File(chkfile, 'a') as fh5:
            if 'mol' not in fh5:
                fh5['mol'] = mol.dumps()
    else:
        save_mol(mol, chkfile)

    scf_dic = {'e_tot'    : e_tot,
               'mo_energy': mo_energy,
               'mo_occ'   : mo_occ,}
    save(chkfile, 'scf', scf_dic)

    # save mo_coeff only if incore mode
    if isinstance(mo_coeff, list):
        nkpts = len(mo_coeff)
        for k in range(nkpts):
            save(chkfile, 'mo_coeff/%d'%k, mo_coeff[k])