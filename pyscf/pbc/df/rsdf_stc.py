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
#
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#         Gengzhi Yang <genzyang17@gmail.com>
#


import numpy as np

from pyscf.lib import logger

from .rsdf import RSGDF, _RSGDFBuilder


def density_fit_j(mf, auxbasis=None, mesh=None, with_df=None):
    '''Generte density-fitting SCF object

    Args:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.  If auxbasis is
            None, auxiliary basis based on AO basis (if possible) or
            even-tempered Gaussian basis will be used.
        mesh : tuple
            number of grids in each direction
        with_df : DF object
    '''
    from pyscf.pbc.scf.hf import KohnShamDFT
    from pyscf.df.addons import predefined_auxbasis
    from pyscf.pbc.df import df
    from pyscf.pbc.scf.khf import KSCF

    if isinstance(mf, KSCF):
        kpts = mf.kpts
    else:
        kpts = numpy.reshape(mf.kpt, (1,3))

    cell = mf.cell
    if auxbasis is None and isinstance(cell.basis, str):
        if isinstance(mf, KohnShamDFT):
            xc = mf.xc
        else:
            xc = 'HF'
        if xc == 'LDA,VWN':
            # This is likely the default xc setting of a KS instance.
            # Postpone the auxbasis assignment to with_df.build().
            auxbasis = None
        else:
            auxbasis = predefined_auxbasis(cell, cell.basis, xc)
    with_df = df.DF(cell, kpts)
    with_df.max_memory = mf.max_memory
    with_df.stdout = mf.stdout
    with_df.verbose = mf.verbose
    with_df.auxbasis = auxbasis
    if mesh is not None:
        with_df.mesh = mesh
    with_df._j_only = True

    return with_df


def density_fit(mf, auxbasis=None, mesh=None, with_df=None, exxdiv='vcut_ws', omega_dot_Rc=4.,
                Rc_type='ws'):
    '''Generate density-fitting SCF object

    Args:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.  If auxbasis is
            None, auxiliary basis based on AO basis (if possible) or
            even-tempered Gaussian basis will be used.
        mesh : tuple
            number of grids in each direction
        with_df : DF object
    '''
    from pyscf.pbc.df import rsdf
    from pyscf.pbc.scf.khf import KSCF

    if isinstance(mf, KSCF):
        kpts = mf.kpts
    else:
        kpts = np.reshape(mf.kpt, (1,3))

    if with_df is None:
        kpts = getattr(kpts, 'kpts', kpts)
        with_df = RSGDF_STC(mf.cell, kpts)
        with_df.exxdiv = exxdiv
        with_df.omega_dot_Rc = omega_dot_Rc
        with_df.Rc_type = Rc_type
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis
        if mesh is not None:
            with_df.mesh = mesh

    mf = mf.copy()
    mf.with_df = with_df
    mf._eri = None

    # add with_df_j for j_build
    mf.with_df.with_df_j = density_fit_j(mf, auxbasis, mesh)

    return mf


class RSGDF_STC(RSGDF):
    omega_dot_Rc = 4.
    exxdiv = 'vcut_ws'
    Rc_type = 'ws'  # inradius of WS; alternative is 'sph'
    with_df_j = None

    def dump_flags(self, verbose=None):
        RSGDF.dump_flags(self, verbose)

        log = logger.new_logger(self, verbose)
        if log.verbose < logger.INFO:
            return self

        log.info('exxdiv= %s', self.exxdiv)
        log.info('omega_dot_Rc= %.15g', self.omega_dot_Rc)
        log.info('Rc_type= %s', self.Rc_type)

        return self

    def _rs_build(self):
        cell = self.cell
        nkpts = len(self.kpts)
        if self.Rc_type.lower() == 'sph':
            Rc = (3*nkpts*cell.vol/(4*np.pi))**(1./3)
        elif self.Rc_type.lower() == 'ws':
            from pyscf.pbc.lo.base import get_kmesh
            from .fft_stc import ws_inradius
            kmesh = get_kmesh(self.cell, self.kpts)
            logger.warn(self, 'Using kmesh= %s to calculate WS-inradius Rc', kmesh)
            Rc = ws_inradius(cell.lattice_vectors(), kmesh)
        else:
            raise NotImplementedError
        self.omega = self.omega_j2c = self.omega_dot_Rc / Rc

        RSGDF._rs_build(self)

    def _make_j3c(self, cell=None, auxcell=None, kptij_lst=None, cderi_file=None):
        if cell is None: cell = self.cell
        if auxcell is None: auxcell = self.auxcell
        if cderi_file is None: cderi_file = self._cderi_to_save

        if self.kpts_band is None:
            kpts_union = self.kpts
        else:
            kpts_union = unique(np.vstack([self.kpts, self.kpts_band]))[0]
        dfbuilder = _RSGDFBuilder_STC(cell, auxcell, kpts_union)
        dfbuilder.__dict__.update(self.__dict__)
        dfbuilder.kpts = kpts_union
        j_only = self._j_only or len(kpts_union) == 1
        dfbuilder.make_j3c(cderi_file, j_only=j_only, dataname=self._dataname,
                           kptij_lst=kptij_lst)

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None and omega != 0:  # J/K for RSH functionals
            raise NotImplementedError

        from pyscf.pbc.df.aft import _check_kpts
        from pyscf.pbc.df import df_jk
        kpts, is_single_kpt = _check_kpts(self, kpts)

        # if is_single_kpt:
        #     return df_jk.get_jk(self, dm, hermi, kpts[0], kpts_band, with_j,
        #                         with_k, exxdiv)

        vj = vk = None
        if with_k:
            vk = df_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
        if with_j:
            vj = df_jk.get_j_kpts(self.with_df_j, dm, hermi, kpts, kpts_band)

        return vj, vk


class _RSGDFBuilder_STC(_RSGDFBuilder):
    exxdiv = 'vcut_ws'

    def weighted_coulG(self, kpt=np.zeros(3), exx=None, mesh=None, omega=None):
        '''Weighted regular Coulomb kernel'''
        from pyscf.pbc import tools as pbctools
        from pyscf.pbc.tools.pbc import _Gv_wrap_around

        if exx is None: exx = self.exxdiv

        cell = self.cell
        if mesh is None:
            mesh = self.mesh
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        coulG = pbctools.get_coulG(cell, kpt, exx, self, mesh, Gv, omega=None)

        # smooth modification
        if abs(kpt).sum() > 1e-9:
            kG = _Gv_wrap_around(cell, Gv, kpt, mesh)
        else:
            kG = Gv
        absG2 = np.einsum('gi,gi->g', kG, kG)
        v0 = coulG[absG2==0].copy()
        coulG *= np.exp(-absG2*0.25/omega**2.)
        coulG[absG2==0] = v0 + np.pi/omega**2.

        coulG *= kws
        return coulG


RSDF_STC = RSGDF_STC
