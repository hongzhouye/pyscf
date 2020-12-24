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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Density fitting

Divide the 3-center Coulomb integrals to two parts.  Compute the local
part in real space, long range part in reciprocal space.

Note when diffuse functions are used in fitting basis, it is easy to cause
linear dependence (non-positive definite) issue under PBC.

Ref:
J. Chem. Phys. 147, 164119 (2017)
'''

import os
import time
import copy
import warnings
import tempfile
import numpy
import h5py
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.df import addons
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.pbc.gto.cell import _estimate_rcut
from pyscf.pbc import tools
from pyscf.pbc.df import outcore
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import aft
from pyscf.pbc.df import df
from pyscf.pbc.df import df_jk
from pyscf.pbc.df import df_ao2mo
from pyscf.pbc.df.aft import estimate_eta, get_nuc
from pyscf.pbc.df.df_jk import zdotCN
from pyscf.pbc.lib.kpts_helper import (is_zero, gamma_point, member, unique,
                                       KPT_DIFF_TOL)
from pyscf.pbc.df.aft import _sub_df_jk_
from pyscf import __config__

from pyscf.pbc.df.df import fuse_auxcell

LINEAR_DEP_THR = getattr(__config__, 'pbc_df_df_DF_lindep', 1e-9)
LONGRANGE_AFT_TURNOVER_THRESHOLD = 2.5


# kpti == kptj: s2 symmetry
# kpti == kptj == 0 (gamma point): real
def _make_j3c(mydf, cell, auxcell, kptij_lst, cderi_file):
    t1 = (time.clock(), time.time())
    log = logger.Logger(mydf.stdout, mydf.verbose)
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    fused_cell, fuse = fuse_auxcell(mydf, auxcell)

    from pyscf.pbc.tools import k2gamma
    bvk_kmesh = k2gamma.kpts_to_kmesh(cell, mydf.kpts)

    # The ideal way to hold the temporary integrals is to store them in the
    # cderi_file and overwrite them inplace in the second pass.  The current
    # HDF5 library does not have an efficient way to manage free space in
    # overwriting.  It often leads to the cderi_file ~2 times larger than the
    # necessary size.  For now, dumping the DF integral intermediates to a
    # separated temporary file can avoid this issue.  The DF intermediates may
    # be terribly huge. The temporary file should be placed in the same disk
    # as cderi_file.
    swapfile = tempfile.NamedTemporaryFile(dir=os.path.dirname(cderi_file))
    fswap = lib.H5TmpFile(swapfile.name)
    # Unlink swapfile to avoid trash
    swapfile = None

    outcore._aux_e2_bvk(cell, fused_cell, fswap, 'int3c2e', aosym='s2',
                        kptij_lst=kptij_lst, dataname='j3c-junk',
                        max_memory=max_memory, bvk_kmesh=bvk_kmesh)
    t1 = log.timer_debug1('3c2e', *t1)

    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    mesh = mydf.mesh
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    b = cell.reciprocal_vectors()
    gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)

    log.debug('Num uniq kpts %d', len(uniq_kpts))
    log.debug2('uniq_kpts %s', uniq_kpts)
    # j2c ~ (-kpt_ji | kpt_ji)
    j2c = fused_cell.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)

    max_memory = max(2000, mydf.max_memory - lib.current_memory()[0])
    blksize = max(2048, int(max_memory*.5e6/16/fused_cell.nao_nr()))
    log.debug2('max_memory %s (MB)  blocksize %s', max_memory, blksize)
    for k, kpt in enumerate(uniq_kpts):
        coulG = mydf.weighted_coulG(kpt, False, mesh)
        for p0, p1 in lib.prange(0, ngrids, blksize):
            aoaux = ft_ao.ft_ao(fused_cell, Gv[p0:p1], None, b, gxyz[p0:p1], Gvbase, kpt).T
            LkR = numpy.asarray(aoaux.real, order='C')
            LkI = numpy.asarray(aoaux.imag, order='C')
            aoaux = None

            if is_zero(kpt):  # kpti == kptj
                j2c[k][naux:] -= lib.ddot(LkR[naux:]*coulG[p0:p1], LkR.T)
                j2c[k][naux:] -= lib.ddot(LkI[naux:]*coulG[p0:p1], LkI.T)
                j2c[k][:naux,naux:] = j2c[k][naux:,:naux].T
            else:
                j2cR, j2cI = zdotCN(LkR[naux:]*coulG[p0:p1],
                                    LkI[naux:]*coulG[p0:p1], LkR.T, LkI.T)
                j2c[k][naux:] -= j2cR + j2cI * 1j
                j2c[k][:naux,naux:] = j2c[k][naux:,:naux].T.conj()
            LkR = LkI = None
        fswap['j2c/%d'%k] = fuse(fuse(j2c[k]).T).T
    j2c = coulG = None

    def cholesky_decomposed_metric(uniq_kptji_id):
        j2c = numpy.asarray(fswap['j2c/%d'%uniq_kptji_id])
        j2c_negative = None
        try:
            j2c = scipy.linalg.cholesky(j2c, lower=True)
            j2ctag = 'CD'
        except scipy.linalg.LinAlgError:
            #msg =('===================================\n'
            #      'J-metric not positive definite.\n'
            #      'It is likely that mesh is not enough.\n'
            #      '===================================')
            #log.error(msg)
            #raise scipy.linalg.LinAlgError('\n'.join([str(e), msg]))
            w, v = scipy.linalg.eigh(j2c)
            log.debug('DF metric linear dependency for kpt %s', uniq_kptji_id)
            log.debug('cond = %.4g, drop %d bfns',
                      w[-1]/w[0], numpy.count_nonzero(w<mydf.linear_dep_threshold))
            v1 = v[:,w>mydf.linear_dep_threshold].conj().T
            v1 /= numpy.sqrt(w[w>mydf.linear_dep_threshold]).reshape(-1,1)
            j2c = v1
            if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
                idx = numpy.where(w < -mydf.linear_dep_threshold)[0]
                if len(idx) > 0:
                    j2c_negative = (v[:,idx]/numpy.sqrt(-w[idx])).conj().T
            w = v = None
            j2ctag = 'eig'
        return j2c, j2c_negative, j2ctag

    feri = h5py.File(cderi_file, 'w')
    feri['j3c-kptij'] = kptij_lst
    nsegs = len(fswap['j3c-junk/0'])
    def make_kpt(uniq_kptji_id, cholesky_j2c):
        kpt = uniq_kpts[uniq_kptji_id]  # kpt = kptj - kpti
        log.debug1('kpt = %s', kpt)
        adapted_ji_idx = numpy.where(uniq_inverse == uniq_kptji_id)[0]
        adapted_kptjs = kptjs[adapted_ji_idx]
        nkptj = len(adapted_kptjs)
        log.debug1('adapted_ji_idx = %s', adapted_ji_idx)

        j2c, j2c_negative, j2ctag = cholesky_j2c

        shls_slice = (auxcell.nbas, fused_cell.nbas)
        Gaux = ft_ao.ft_ao(fused_cell, Gv, shls_slice, b, gxyz, Gvbase, kpt)
        wcoulG = mydf.weighted_coulG(kpt, False, mesh)
        Gaux *= wcoulG.reshape(-1,1)
        kLR = Gaux.real.copy('C')
        kLI = Gaux.imag.copy('C')
        Gaux = None

        if is_zero(kpt):  # kpti == kptj
            aosym = 's2'
            nao_pair = nao*(nao+1)//2

            if cell.dimension == 3:
                vbar = fuse(mydf.auxbar(fused_cell))
                ovlp = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=adapted_kptjs)
                ovlp = [lib.pack_tril(s) for s in ovlp]
        else:
            aosym = 's1'
            nao_pair = nao**2

        mem_now = lib.current_memory()[0]
        log.debug2('memory = %s', mem_now)
        max_memory = max(2000, mydf.max_memory-mem_now)
        # nkptj for 3c-coulomb arrays plus 1 Lpq array
        buflen = min(max(int(max_memory*.38e6/16/naux/(nkptj+1)), 1), nao_pair)
        shranges = _guess_shell_ranges(cell, buflen, aosym)
        buflen = max([x[2] for x in shranges])
        # +1 for a pqkbuf
        if aosym == 's2':
            Gblksize = max(16, int(max_memory*.1e6/16/buflen/(nkptj+1)))
        else:
            Gblksize = max(16, int(max_memory*.2e6/16/buflen/(nkptj+1)))
        Gblksize = min(Gblksize, ngrids, 16384)
        pqkRbuf = numpy.empty(buflen*Gblksize)
        pqkIbuf = numpy.empty(buflen*Gblksize)
        # buf for ft_aopair
        buf = numpy.empty(nkptj*buflen*Gblksize, dtype=numpy.complex128)
        def pw_contract(istep, sh_range, j3cR, j3cI):
            bstart, bend, ncol = sh_range
            if aosym == 's2':
                shls_slice = (bstart, bend, 0, bend)
            else:
                shls_slice = (bstart, bend, 0, cell.nbas)

            for p0, p1 in lib.prange(0, ngrids, Gblksize):
                dat = ft_ao.ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym,
                                           b, gxyz[p0:p1], Gvbase, kpt,
                                           adapted_kptjs, bvk_kmesh=bvk_kmesh,
                                           out=buf)
                nG = p1 - p0
                for k, ji in enumerate(adapted_ji_idx):
                    aoao = dat[k].reshape(nG,ncol)
                    pqkR = numpy.ndarray((ncol,nG), buffer=pqkRbuf)
                    pqkI = numpy.ndarray((ncol,nG), buffer=pqkIbuf)
                    pqkR[:] = aoao.real.T
                    pqkI[:] = aoao.imag.T

                    lib.dot(kLR[p0:p1].T, pqkR.T, -1, j3cR[k][naux:], 1)
                    lib.dot(kLI[p0:p1].T, pqkI.T, -1, j3cR[k][naux:], 1)
                    if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                        lib.dot(kLR[p0:p1].T, pqkI.T, -1, j3cI[k][naux:], 1)
                        lib.dot(kLI[p0:p1].T, pqkR.T,  1, j3cI[k][naux:], 1)

            for k, ji in enumerate(adapted_ji_idx):
                if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                    v = fuse(j3cR[k])
                else:
                    v = fuse(j3cR[k] + j3cI[k] * 1j)
                if j2ctag == 'CD':
                    v = scipy.linalg.solve_triangular(j2c, v, lower=True, overwrite_b=True)
                    feri['j3c/%d/%d'%(ji,istep)] = v
                else:
                    feri['j3c/%d/%d'%(ji,istep)] = lib.dot(j2c, v)

                # low-dimension systems
                if j2c_negative is not None:
                    feri['j3c-/%d/%d'%(ji,istep)] = lib.dot(j2c_negative, v)

        with lib.call_in_background(pw_contract) as compute:
            col1 = 0
            for istep, sh_range in enumerate(shranges):
                log.debug1('int3c2e [%d/%d], AO [%d:%d], ncol = %d',
                           istep+1, len(shranges), *sh_range)
                bstart, bend, ncol = sh_range
                col0, col1 = col1, col1+ncol
                j3cR = []
                j3cI = []
                for k, idx in enumerate(adapted_ji_idx):
                    v = numpy.vstack([fswap['j3c-junk/%d/%d'%(idx,i)][0,col0:col1].T
                                      for i in range(nsegs)])
                    # vbar is the interaction between the background charge
                    # and the auxiliary basis.  0D, 1D, 2D do not have vbar.
                    if is_zero(kpt) and cell.dimension == 3:
                        for i in numpy.where(vbar != 0)[0]:
                            v[i] -= vbar[i] * ovlp[k][col0:col1]
                    j3cR.append(numpy.asarray(v.real, order='C'))
                    if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                        j3cI.append(None)
                    else:
                        j3cI.append(numpy.asarray(v.imag, order='C'))
                v = None
                compute(istep, sh_range, j3cR, j3cI)
        for ji in adapted_ji_idx:
            del(fswap['j3c-junk/%d'%ji])

    # Wrapped around boundary and symmetry between k and -k can be used
    # explicitly for the metric integrals.  We consider this symmetry
    # because it is used in the df_ao2mo module when contracting two 3-index
    # integral tensors to the 4-index 2e integral tensor. If the symmetry
    # related k-points are treated separately, the resultant 3-index tensors
    # may have inconsistent dimension due to the numerial noise when handling
    # linear dependency of j2c.
    def conj_j2c(cholesky_j2c):
        j2c, j2c_negative, j2ctag = cholesky_j2c
        if j2c_negative is None:
            return j2c.conj(), None, j2ctag
        else:
            return j2c.conj(), j2c_negative.conj(), j2ctag

    a = cell.lattice_vectors() / (2*numpy.pi)
    def kconserve_indices(kpt):
        '''search which (kpts+kpt) satisfies momentum conservation'''
        kdif = numpy.einsum('wx,ix->wi', a, uniq_kpts + kpt)
        kdif_int = numpy.rint(kdif)
        mask = numpy.einsum('wi->i', abs(kdif - kdif_int)) < KPT_DIFF_TOL
        uniq_kptji_ids = numpy.where(mask)[0]
        return uniq_kptji_ids

    done = numpy.zeros(len(uniq_kpts), dtype=bool)
    for k, kpt in enumerate(uniq_kpts):
        if done[k]:
            continue

        log.debug1('Cholesky decomposition for j2c at kpt %s', k)
        cholesky_j2c = cholesky_decomposed_metric(k)

        # The k-point k' which has (k - k') * a = 2n pi. Metric integrals have the
        # symmetry S = S
        uniq_kptji_ids = kconserve_indices(-kpt)
        log.debug1("Symmetry pattern (k - %s)*a= 2n pi", kpt)
        log.debug1("    make_kpt for uniq_kptji_ids %s", uniq_kptji_ids)
        for uniq_kptji_id in uniq_kptji_ids:
            if not done[uniq_kptji_id]:
                make_kpt(uniq_kptji_id, cholesky_j2c)
        done[uniq_kptji_ids] = True

        # The k-point k' which has (k + k') * a = 2n pi. Metric integrals have the
        # symmetry S = S*
        uniq_kptji_ids = kconserve_indices(kpt)
        log.debug1("Symmetry pattern (k + %s)*a= 2n pi", kpt)
        log.debug1("    make_kpt for %s", uniq_kptji_ids)
        cholesky_j2c = conj_j2c(cholesky_j2c)
        for uniq_kptji_id in uniq_kptji_ids:
            if not done[uniq_kptji_id]:
                make_kpt(uniq_kptji_id, cholesky_j2c)
        done[uniq_kptji_ids] = True

    feri.close()


class BvKGDF(df.GDF):
    '''Gaussian density fitting
    '''
    def __init__(self, cell, kpts=numpy.zeros((1,3))):
        df.GDF.__init__(self, cell, kpts)

    _make_j3c = _make_j3c
