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
#         Hong-Zhou Ye <hzyechem@gmail.com>
#

'''
Pipek-Mezey localization

ref. JCTC 10, 642 (2014); DOI:10.1021/ct401016x
'''

import numpy
from functools import reduce

from pyscf import lib
from pyscf.lib import logger
from pyscf.lo import orth
from pyscf.lo import boys
from pyscf.lo import iao
from pyscf.lo.stability import stability_jacobi, stability_newton
from pyscf import __config__


def atomic_pops(mol, mo_coeff, method='meta_lowdin', kpt=None, proj_data=None, mode=None):
    '''
    Kwargs:
        method : string
            The atomic population projection scheme. It can be mulliken,
            lowdin, meta_lowdin, iao, or becke

    Returns:
        A 3-index tensor [A,i,j] indicates the population of any orbital-pair
        density |i><j| for each species (atom in this case).  This tensor is
        used to construct the population and gradients etc.

        You can customize the PM localization wrt other population metric,
        such as the charge of a site, the charge of a fragment (a group of
        atoms) by overwriting this tensor.  See also the example
        pyscf/examples/loc_orb/40-hubbard_model_PM_localization.py for the PM
        localization of site-based population for hubbard model.
    '''
    method = method.lower().replace('_', '-')
    nmo = mo_coeff.shape[1]

    def proj_orth(mo_coeff, proj_coeff, offset_nr_by_atom):
        csc = lib.dot(proj_coeff.conj().T, mo_coeff)

        if mode == 'pop':
            proj = numpy.empty((mol.natm,nmo), dtype=numpy.float64)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                proj[i] = (abs(csc[p0:p1])**2).sum(axis=0)
        else:
            proj = numpy.empty((mol.natm,nmo,nmo), dtype=mo_coeff.dtype)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                lib.dot(csc[p0:p1].conj().T, csc[p0:p1], c=proj[i])
        return proj

    def proj_biorth(mo_coeff, proj_coeff, projtild_coeff, offset_nr_by_atom):
        csc = lib.dot(proj_coeff.conj().T, mo_coeff)
        csctild = lib.dot(projtild_coeff.conj().T, mo_coeff)

        if mode == 'pop':
            proj = numpy.empty((mol.natm,nmo), dtype=numpy.float64)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                proj[i] = (csc[p0:p1].conj()*csctild[p0:p1]).sum(axis=0).real
        else:
            proj = numpy.empty((mol.natm,nmo,nmo), dtype=mo_coeff.dtype)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                lib.dot(csc[p0:p1].conj().T, csctild[p0:p1], c=proj[i], alpha=0.5)
                proj[i] += proj[i].conj().T
        return proj

    if proj_data is None:
        proj_data = get_proj_data(mol, mo_coeff, method, kpt)

    if method == 'becke':
        charge_matrices = proj_data

        if mode == 'pop':
            proj = numpy.empty((mol.natm,nmo), dtype=numpy.float64)
            for i in range(mol.natm):
                proj[i] = lib.einsum('mi,mn,ni->i', mo_coeff.conj(), charge_matrices[i],
                                     mo_coeff).real
        else:
            proj = numpy.empty((mol.natm,nmo,nmo), dtype=mo_coeff.dtype)
            for i in range(mol.natm):
                proj[i] = reduce(lib.dot, (mo_coeff.conj().T, charge_matrices[i], mo_coeff))

    elif method == 'mulliken':
        s = get_ovlp(mol, kpt)
        csc = mo_coeff
        csctild = lib.dot(s, mo_coeff)
        if mode == 'pop':
            proj = numpy.empty((mol.natm,nmo), dtype=numpy.float64)
            for i, (b0, b1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
                proj[i] = (csc[p0:p1].conj()*csctild[p0:p1]).sum(axis=0).real
        else:
            proj = numpy.empty((mol.natm,nmo,nmo), dtype=mo_coeff.dtype)
            for i, (b0, b1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
                proj[i] = lib.dot(csc[p0:p1].conj().T, csctild[p0:p1], c=proj[i], alpha=0.5)
                proj[i] += proj[i].conj().T

    elif method in ('lowdin', 'meta-lowdin'):
        proj_coeff, offset_nr_by_atom = proj_data
        proj = proj_orth(mo_coeff, proj_coeff, offset_nr_by_atom)

    elif method == 'iao-biorth':
        proj_coeff, projtild_coeff, offset_nr_by_atom = proj_data
        proj = proj_biorth(mo_coeff, proj_coeff, projtild_coeff, offset_nr_by_atom)

    elif method in ('iao', 'ibo'):  # Why is 'ibo' the same as 'iao'...?
        proj_coeff, offset_nr_by_atom = proj_data
        proj = proj_orth(mo_coeff, proj_coeff, offset_nr_by_atom)

    else:
        raise KeyError('method = %s' % method)

    return proj


def get_ovlp(mol, kpt=None):
    if getattr(mol, 'pbc_intor', None):  # whether mol object is a cell
        s = mol.pbc_intor('int1e_ovlp', hermi=1, kpt=kpt)
    else:
        s = mol.intor_symmetric('int1e_ovlp')
    return s


def becke_charge_matrices(mol):
    from pyscf.dft import gen_grid
    # Call DFT to initialize grids and numint objects
    mf = mol.RKS()
    grids = mf.grids
    ni = mf._numint

    if not isinstance(grids, gen_grid.Grids):
        raise NotImplementedError('PM becke scheme for PBC systems')

    # The atom-wise Becke grids (without concatenated to a vector of grids)
    coords, weights = grids.get_partition(mol, concat=False)

    charge_matrices = []
    for i in range(mol.natm):
        ao = ni.eval_ao(mol, coords[i], deriv=0)
        aow = numpy.einsum('pi,p->pi', ao, weights[i])
        charge_matrices.append(lib.dot(aow.conj().T, ao))

    return charge_matrices


def get_proj_data(mol, mo_coeff, method, kpt):

    method = method.lower().replace('_', '-')

    if method == 'becke':
        proj_data = becke_charge_matrices(mol)

    elif method == 'mulliken':
        proj_data = None

    elif method in ('lowdin', 'meta-lowdin'):
        s = get_ovlp(mol, kpt)
        proj_coeff = orth.orth_ao(mol, method, 'ANO', s=s, adjust_phase=False)
        proj_coeff = lib.dot(s, proj_coeff)
        proj_data = (proj_coeff, mol.offset_nr_by_atom())

    elif method in ('iao', 'ibo', 'iao-biorth'):
        s = get_ovlp(mol, kpt)
        if kpt is None:
            iao_coeff = iao.iao(mol, mo_coeff)
        else:
            iao_coeff = iao.iao(mol, [mo_coeff], kpts=[kpt])[0]
        iao_mol = iao.reference_mol(mol)

        if method == 'iao-biorth':
            ovlp = reduce(lib.dot, (iao_coeff.conj().T, s, iao_coeff))
            iaotild_coeff = numpy.asarray(numpy.linalg.solve(ovlp,
                                          iao_coeff.conj().T).conj().T, order='C')
            proj_coeff = lib.dot(s, iao_coeff)
            projtild_coeff = lib.dot(s, iaotild_coeff)
            proj_data = (proj_coeff, projtild_coeff, iao_mol.offset_nr_by_atom())
        else:
            iao_coeff = orth.vec_lowdin(iao_coeff, s)
            proj_coeff = lib.dot(s, iao_coeff)
            proj_data = (proj_coeff, iao_mol.offset_nr_by_atom())

    else:
        raise KeyError('method = %s' % method)

    return proj_data


class PipekMezey(boys.OrbitalLocalizer):
    '''The Pipek-Mezey localization optimizer that maximizes the orbital
    population

    Args:
        mol : Mole object

    Kwargs:
        mo_coeff : size (N,N) numpy.array
            The orbital space to localize for PM localization.
            When initializing the localization optimizer ``bopt = PM(mo_coeff)``,

            Note these orbitals ``mo_coeff`` may or may not be used as initial
            guess, depending on the attribute ``.init_guess`` . If ``.init_guess``
            is set to None, the ``mo_coeff`` will be used as initial guess. If
            ``.init_guess`` is 'atomic', a few atomic orbitals will be
            constructed inside the space of the input orbitals and the atomic
            orbitals will be used as initial guess.

            Note when calling .kernel(orb) method with a set of orbitals as
            argument, the orbitals will be used as initial guess regardless of
            the value of the attributes .mo_coeff and .init_guess.

    Attributes for PM class:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`.
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`.
        conv_tol : float
            Converge threshold.  Default 1e-6
        conv_tol_grad : float
            Converge threshold for orbital rotation gradients.  Default 1e-3
        max_cycle : int
            The max. number of macro iterations. Default 100
        max_iters : int
            The max. number of iterations in each macro iteration. Default 20
        max_stepsize : float
            The step size for orbital rotation.  Small step (0.005 - 0.05) is preferred.
            Default 0.03.
        init_guess : str or None
            Initial guess for optimization. If set to None, orbitals defined
            by the attribute .mo_coeff will be used as initial guess. If set
            to 'atomic', atomic orbitals will be used as initial guess.
            Default 'atomic'
        pop_method : str
            How the orbital population is calculated, see JCTC 10, 642
            (2014) for discussion. Options are:
            - 'meta-lowdin' (default) as defined in JCTC 10, 3784 (2014)
            - 'mulliken' original Pipek-Mezey scheme, JCP 90, 4916 (1989)
            - 'lowdin' Lowdin charges, JCTC 10, 642 (2014)
            - 'iao' or 'ibo' intrinsic atomic orbitals, JCTC 9, 4384 (2013)
            - 'becke' Becke charges, JCTC 10, 642 (2014)
            The IAO and Becke charges do not depend explicitly on the
            basis set, and have a complete basis set limit [JCTC 10,
            642 (2014)].
        exponent : int
            The power to define norm. It can be 2 or 4. Default 2.

    Saved results

        mo_coeff : ndarray
            Localized orbitals

    '''


    pop_method = getattr(__config__, 'lo_pipek_PM_pop_method', 'meta_lowdin')
    conv_tol = getattr(__config__, 'lo_pipek_PM_conv_tol', 1e-6)
    exponent = getattr(__config__, 'lo_pipek_PM_exponent', 2)  # any integer >= 2

    _keys = {'pop_method', 'conv_tol', 'exponent', 'kpt', '_proj_data'}

    def __init__(self, mol, mo_coeff=None, pop_method=None, kpt=None):
        boys.OrbitalLocalizer.__init__(self, mol, mo_coeff)
        self.maximize = True
        if pop_method is not None:
            self.pop_method = pop_method
        self.kpt = kpt
        self._proj_data = None

    def dump_flags(self, verbose=None):
        boys.OrbitalLocalizer.dump_flags(self, verbose)
        logger.info(self, 'pop_method = %s',self.pop_method)
        logger.info(self, 'exponent = %s',self.exponent)

    def get_proj_data(self, mol=None, mo_coeff=None, method=None, kpt=None):
        if mol is None: mol = self.mol
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if method is None: method = self.pop_method.lower().replace('_', '-')
        if kpt is None: kpt = self.kpt

        log = logger.new_logger(self, verbose=self.verbose-1)
        cput0 = (logger.process_clock(), logger.perf_counter())

        proj_data = get_proj_data(mol, mo_coeff, method, kpt)

        log.timer('get_proj_data', *cput0)

        return proj_data

    def gen_g_hop(self, u=None):
        log = logger.new_logger(self, verbose=self.verbose-1)
        cput0 = (logger.process_clock(), logger.perf_counter())

        exponent = self.exponent
        projR = self.atomic_pops(u).real    # real rotations only need proj.real
        pop = lib.einsum('xii->xi', projR)
        popexp1 = pop**(exponent-1)
        popexp2 = pop**(exponent-2)

        # gradient
        g = self.get_grad(proj=projR)

        # hessian diagonal
        g1 = lib.einsum('xi,xi->i', popexp1, pop)
        g2 = lib.einsum('xi,xj->ij', popexp1, pop)
        h_diag  = 2 * exponent * (g1[:,None] - g2)
        g1 = lib.einsum('xi,xij->ij', popexp2, projR**2)
        h_diag += -4 * exponent * (exponent-1) * g1
        h_diag = self.pack_uniq_var(h_diag + h_diag.T)

        # hessian vector product
        G = lib.einsum('xi,xij->ij', popexp1, projR)

        mem_avail = self.mol.max_memory - lib.current_memory()[0]
        if mem_avail * 0.5 > self.norb**3 * 8/1024**2:
            QP = lib.einsum('xj,xil->ilj', popexp1, projR)
            def h_op(x):
                x = self.unpack_uniq_var(x)

                # contributions from disconnected term
                j0 = popexp2 * lib.einsum('xik,ki->xi', projR, x)
                j1 = lib.einsum('xi,xij->ij', j0, projR)
                hx = 4 * exponent * (exponent-1) * j1

                # contributions symmetric connected terms
                j1 = lib.einsum('ilj,lj->ij', QP, x)
                hx += -2 * exponent * j1

                # contributions from asymmetric connected terms
                j1 = numpy.dot(G, x)
                j1 += numpy.dot(x, G)
                hx += exponent * j1

                return self.pack_uniq_var(hx - hx.T)

        else:
            def h_op(x):
                x = self.unpack_uniq_var(x)

                projx = lib.einsum('xik,kj->xij', projR, x)

                # contributions from disconnected term
                j0 = popexp2 * lib.einsum('xii->xi', projx)
                j1 = lib.einsum('xi,xij->ij', j0, projR)
                hx = 4 * exponent * (exponent-1) * j1

                # contributions symmetric connected terms
                j1 = lib.einsum('xj,xij->ij', popexp1, projx)
                hx += -2 * exponent * j1

                # contributions from asymmetric connected terms
                j1 = numpy.dot(G, x)
                j1 += numpy.dot(x, G)
                hx += exponent * j1

                return self.pack_uniq_var(hx - hx.T)

        log.timer('gen_g_hop', *cput0)

        return g, h_op, h_diag

    def get_grad(self, u=None, proj=None):
        if proj is None:
            proj = self.atomic_pops(u)

        exponent = self.exponent
        popexp1 = lib.einsum('xii->xi', proj.real)**(exponent-1)
        g = lib.einsum('xi,xij->ij', popexp1, proj.real)
        return 2 * exponent * self.pack_uniq_var(g - g.T)

    def cost_function(self, u=None, mode='pop'):
        if mode == 'pop':
            pop = self.atomic_pops(u, mode=mode)
            return (pop**self.exponent).sum()
        else:
            proj = self.atomic_pops(u)
            return (lib.einsum('xii->xi', proj.real)**self.exponent).sum()

    @lib.with_doc(atomic_pops.__doc__)
    def atomic_pops(self, u=None, mode=None):
        mol = self.mol
        mo_coeff = self.rotate_orb(u)
        method = self.pop_method

        return atomic_pops(mol, mo_coeff, method, kpt=self.kpt, proj_data=self._proj_data,
                           mode=mode)

    def kernel(self, mo_coeff=None, callback=None, verbose=None):
        self._proj_data = self.get_proj_data()
        mo_coeff = boys.kernel(self, mo_coeff, callback, verbose)
        self._proj_data = None

        return mo_coeff

    def stability_jacobi(self, verbose=None, return_status=False):
        self._proj_data = self.get_proj_data()
        res = stability_jacobi(self, verbose=verbose, return_status=return_status)
        self._proj_data = None

        return res

    def stability(self, verbose=None, return_status=False):
        self._proj_data = self.get_proj_data()
        res = stability_newton(self, verbose=verbose, return_status=return_status)
        self._proj_data = None

        return res


PM = Pipek = PipekMezey


@lib.with_doc(PipekMezey.__doc__)
class PipekMezeyComplex(PipekMezey, boys.OrbitalLocalizerComplex):
    def __init__(self, mol, mo_coeff=None, pop_method=None, kpt=None):
        boys.OrbitalLocalizerComplex.__init__(self, mol, mo_coeff)
        self.maximize = True
        if pop_method is not None:
            self.pop_method = pop_method
        self.kpt = kpt
        self._proj_data = None

    def gen_g_hop(self, u=None):
        exponent = self.exponent
        proj = self.atomic_pops(u)
        pop = lib.einsum('xii->xi', proj.real)
        popexp1 = pop**(exponent-1)
        popexp2 = pop**(exponent-2)

        # gradient
        g = self.get_grad(proj=proj)

        # hessian diagonal
        g1 = lib.einsum('xi->i', pop**exponent)
        g2 = lib.einsum('xi,xj->ij', popexp1, pop)
        h_diag = 2*exponent * (g1[:,None] - g2) * (1 + 1j)
        g1 = lib.einsum('xi,xij->ij', popexp2, proj.real**2)
        g2 = lib.einsum('xi,xij->ij', popexp2, proj.imag**2)
        h_diag += -4*exponent*(exponent-1) * (g1 + g2 * 1j)
        h_diag = self.pack_uniq_var(h_diag + h_diag.T)

        # hessian vector product
        G = lib.einsum('xi,xij->ij', popexp1, proj)

        mem_avail = self.mol.max_memory - lib.current_memory()[0]
        if mem_avail * 0.5 > self.norb**3 * 16/1024**2:
            QP = lib.einsum('xj,xil->ilj', popexp1, proj)
            def h_op(x):
                x = self.unpack_uniq_var(x)

                # contributions from disconnected term
                j0 = popexp2 * lib.einsum('xik,ki->xi', proj, x).real
                j1 = lib.einsum('xi,xij->ij', j0, proj)
                hx = 4 * exponent * (exponent-1) * j1.astype(numpy.complex128)

                # contributions symmetric connected terms
                j1 = lib.einsum('ilj,lj->ij', QP, x)
                hx += -2 * exponent * j1

                # contributions from asymmetric connected terms
                j1 = numpy.dot(G, x)
                j1 += numpy.dot(x, G)
                hx += exponent * j1

                return self.pack_uniq_var(hx - hx.conj().T)

        else:
            def h_op(x):
                x = self.unpack_uniq_var(x)

                projx = lib.einsum('xik,kj->xij', proj, x)

                # contributions from disconnected term
                j0 = popexp2 * lib.einsum('xii->xi', projx.real)
                j1 = lib.einsum('xi,xij->ij', j0, proj)
                hx = 4 * exponent * (exponent-1) * j1.astype(numpy.complex128)

                # contributions symmetric connected terms
                j1 = lib.einsum('xj,xij->ij', popexp1, projx)
                hx += -2 * exponent * j1

                # contributions from asymmetric connected terms
                # j1 = lib.einsum('xi,xij->ij', popexp1, projx)
                j1 = numpy.dot(G, x)
                j1 += numpy.dot(x, G)
                hx += exponent * j1

                return self.pack_uniq_var(hx - hx.conj().T)

        return g, h_op, h_diag

    def get_grad(self, u=None, proj=None):
        if proj is None:
            proj = self.atomic_pops(u)

        exponent = self.exponent
        popexp1 = lib.einsum('xii->xi', proj.real)**(exponent-1)
        g = lib.einsum('xi,xij->ij', popexp1, proj)
        return 2 * exponent * self.pack_uniq_var(g - g.conj().T)


PMComplex = PipekComplex = PipekMezeyComplex


if __name__ == '__main__':
    from pyscf.pbc import gto, scf

    atom = '''
    O          0.00000        0.00000        0.11779
    H          0.00000        0.75545       -0.47116
    H          0.00000       -0.75545       -0.47116
    '''
    basis = 'ccpvdz'
    a = numpy.eye(3) * 4

    # atom = '''
    # H      1.2194     -0.1652      2.1600
    # C      0.6825     -0.0924      1.2087
    # C     -0.7075     -0.0352      1.1973
    # H     -1.2644     -0.0630      2.1393
    # C     -1.3898      0.0572     -0.0114
    # H     -2.4836      0.1021     -0.0204
    # C     -0.6824      0.0925     -1.2088
    # H     -1.2194      0.1652     -2.1599
    # C      0.7075      0.0352     -1.1973
    # H      1.2641      0.0628     -2.1395
    # C      1.3899     -0.0572      0.0114
    # H      2.4836     -0.1022      0.0205
    # '''
    # basis = 'ccpvdz'
    # a = numpy.eye(3) * 7

    cell = gto.M(atom=atom, a=a, basis=basis).set(verbose=4)

    kpt = cell.make_kpts([1,1,1], scaled_center=[0.37, 0.21, 0.85])[0]
    # kpt = None

    mf = scf.RHF(cell, kpt=kpt).rs_density_fit()
    mf.kernel()

    mo = mf.mo_coeff[:,mf.mo_occ>1e-6]
    mlo = PM(cell, mo, kpt=kpt)
    mlo.kernel()

    # stability check
    while True:
        mo, stable = mlo.stability_jacobi(return_status=True)
        # mo, stable = mlo.stability(return_status=True)
        if stable:
            break
        mlo.kernel(mo)

    mo = mf.mo_coeff[:,mf.mo_occ>1e-6]
    mlo = PM(cell, mo, kpt=kpt)
    mlo.algorithm = 'bfgs'
    mlo.kernel()

    # stability check
    while True:
        # mo, stable = mlo.stability_jacobi(return_status=True)
        mo, stable = mlo.stability(return_status=True)
        if stable:
            break
        mlo.kernel(mo)

    mlo = PMComplex(cell, mo, kpt=kpt)
    mlo.kernel()

    # stability check
    while True:
        mo, stable = mlo.stability_jacobi(return_status=True)
        # mo, stable = mlo.stability(return_status=True)
        if stable:
            break
        mlo.kernel(mo)


    # from pyscf import gto, scf
    #
    # mol = gto.Mole()
    # mol.atom = '''
    # O          0.00000        0.00000        0.11779
    # H          0.00000        0.75545       -0.47116
    # H          0.00000       -0.75545       -0.47116
    # '''
    # mol.basis = 'ccpvdz'
    # mol.build()
    # mf = scf.RHF(mol).run()
    #
    # def findiff_grad(func, x, delta=1e-4):
    #     ''' Finite-difference gradient
    #     '''
    #     x = numpy.asarray(x)
    #     n = x.size
    #     g = numpy.zeros_like(x)
    #     for i in range(n):
    #         dx = numpy.zeros_like(x)
    #         dx[i] = delta*0.5
    #         g[i] = (func(x+dx) - func(x-dx)) / delta
    #     return g
    #
    # def findiff_hess(func, x, delta=1e-4):
    #     ''' Finite-difference Hessian
    #     '''
    #     x = numpy.asarray(x)
    #     n = x.size
    #     h = numpy.zeros((n,n), dtype=x.dtype)
    #     for i in range(n):
    #         dxi = numpy.zeros_like(x)
    #         dxi[i] = delta*0.5
    #         for j in range(i+1):
    #             dxj = numpy.zeros_like(x)
    #             dxj[j] = delta*0.5
    #             hij = (func(x+dxi+dxj) + func(x-dxi-dxj) - func(x+dxi-dxj) - func(x-dxi+dxj)) / delta**2
    #             h[i,j] = h[j,i] = hij
    #     return h
    #
    # mo0 = mf.mo_coeff[:,mf.mo_occ>1e-6]
    # # mo0 = mo0 + numpy.random.rand(*mo0.shape) * (0.1)
    # mo0 = mo0 + numpy.random.rand(*mo0.shape) * (0.1+0.1j)
    # # mlo = PM(mol, mo0)
    # mlo = PMComplex(mol, mo0)
    #
    # g, h_op, h_diag = mlo.gen_g_hop()
    # x = mlo.zero_uniq_var()
    # h = numpy.zeros((mlo.pdim,mlo.pdim))
    # for i in range(mlo.pdim):
    #     x[i] = 1
    #     h[:,i] = h_op(x)
    #     x[i] = 0
    #
    # # finite difference
    # def func(x):
    #     u = mlo.extract_rotation(x)
    #     return -mlo.cost_function(u)
    #
    # g1 = findiff_grad(func, x)
    # g_err = abs(g-g1).max()
    # print(f'Grad err: {g_err:.3e}')
    #
    # h1 = findiff_hess(func, x)
    # h_err = abs(h-h1).max()
    # print(f'Hess err: {h_err:.3e}')
    #
    # hd_err = abs(h_diag-numpy.diag(h1)).max()
    # print(f'Hess-diag err: {hd_err:.3e}')
    #
    # # mlo = PM(mol)
    # # mlo.verbose = 4
    # # mlo.exponent = 2    # integer >= 2
    # # mo0 = mf.mo_coeff[:,mf.mo_occ>1e-6]
    # # mo = mlo.kernel(mo0)
    # # isstable, mo1 = mlo.stability_jacobi()
    # # if not isstable:
    # #     mo = mlo.kernel(mo1)
    # #     isstable, mo1 = mlo.stability_jacobi()
    # #     assert( isstable )
