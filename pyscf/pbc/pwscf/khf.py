""" Hartree-Fock in the Plane Wave Basis
"""


import time
import copy
import h5py
import tempfile
import numpy as np
import scipy.linalg

from pyscf import lib
from pyscf import __config__
from pyscf.scf import hf as mol_hf
from pyscf.scf import chkfile as mol_chkfile
from pyscf.pbc.pwscf import chkfile
from pyscf.pbc import gto, scf, tools
from pyscf.pbc.pwscf import pw_helper
from pyscf.pbc.pwscf.pw_helper import get_kcomp, set_kcomp
from pyscf.pbc.pwscf import pseudo as pw_pseudo
from pyscf.pbc.pwscf import jk as pw_jk
from pyscf.lib import logger
import pyscf.lib.parameters as param


# TODO
# 1. fractional occupation (for metals)


THR_OCC = 1E-3


def kernel_doubleloop(mf, kpts, C0=None,
            nbandv=0, nbandv_extra=1,
            conv_tol=1.E-6, conv_tol_davidson=1.E-6, conv_tol_band=1e-4,
            max_cycle=100, max_cycle_davidson=10, verbose_davidson=0,
            ace_exx=True, damp_type="anderson", damp_factor=0.3,
            dump_chk=True, conv_check=True, callback=None, **kwargs):
    ''' Kernel function for SCF in a PW basis
        Note:
            This double-loop implementation follows closely the implementation in Quantum ESPRESSO.

        Args:
            C0 (list of numpy arrays):
                A list of nkpts numpy arrays, each of size nocc(k) * Npw.
            nbandv (int):
                How many virtual bands to compute? Default is zero.
            nbandv_extra (int):
                How many extra virtual bands to include to facilitate the convergence of the davidson algorithm for the highest few virtual bands? Default is 1.
    '''

    cput0 = (time.clock(), time.time())

    cell = mf.cell
    nkpts = len(kpts)

    nbando, nbandv_tot, nband, nband_tot = mf.get_nband(nbandv, nbandv_extra)
    logger.info(mf, "Num of occ bands= %s", nbando)
    logger.info(mf, "Num of vir bands= %s", nbandv)
    logger.info(mf, "Num of all bands= %s", nband)
    logger.info(mf, "Num of extra vir bands= %s", nbandv_extra)

    # init guess and SCF chkfile
    tick = np.asarray([time.clock(), time.time()])

    if mf.outcore:
        swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        fswap = lib.H5TmpFile(swapfile.name)
        swapfile = None
        C_ks = fswap.create_group("C_ks")
    else:
        fswap = None
        C_ks = None
    C_ks, mocc_ks = mf.get_init_guess(nvir=nbandv_tot, C0=C0, out=C_ks)

    tock = np.asarray([time.clock(), time.time()])
    mf.scf_summary["t-init"] = tock - tick

    # init E
    mesh = cell.mesh
    Gv = cell.get_Gv(mesh)
    vj_R = mf.get_vj_R(C_ks, mocc_ks, mesh=mesh, Gv=Gv)
    mf.update_pp(C_ks)
    mf.update_k(C_ks, mocc_ks)
    C_ks, moe_ks, mocc_ks = mf.eig_subspace(C_ks, mocc_ks)
    # C_ks, moe_ks, mocc_ks = sort_mo(C_ks, moe_ks, mocc_ks)
    e_tot = mf.energy_tot(C_ks, mocc_ks, moe_ks=moe_ks)
    logger.info(mf, 'init E= %.15g', e_tot)
    mf.dump_moe(moe_ks, mocc_ks, nband=nband)

    scf_conv = False

    if mf.max_cycle <= 0:
        remove_extra_virbands(C_ks, moe_ks, mocc_ks, nbandv_extra)
        return scf_conv, e_tot, moe_ks, C_ks, mocc_ks

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        mol_chkfile.save_mol(cell, mf.chkfile)

    fc_tot = 0
    fc_this = 0
    cput1 = logger.timer(mf, 'initialize pwscf', *cput0)

    chg_conv_tol = 0.1
    for cycle in range(max_cycle):

        if cycle > 0:
            chg_conv_tol = min(chg_conv_tol, max(conv_tol, 0.1*abs(de)))
        conv_tol_davidson = max(conv_tol*0.1, chg_conv_tol*0.01)
        logger.debug(mf, "  Performing charge SCF with conv_tol= %.3g conv_tol_davidson= %.3g", chg_conv_tol, conv_tol_davidson)

        # charge SCF
        chg_scf_conv, fc_this, C_ks, chg_moe_ks, chg_mocc_ks, chg_e_tot = \
                            mf.kernel_charge(
                                C_ks, mocc_ks, kpts, nband, mesh=mesh, Gv=Gv,
                                max_cycle=max_cycle, conv_tol=chg_conv_tol,
                                max_cycle_davidson=max_cycle_davidson,
                                conv_tol_davidson=conv_tol_davidson,
                                verbose_davidson=verbose_davidson,
                                damp_type=damp_type, damp_factor=damp_factor,
                                vj_R=vj_R,
                                last_hf_e=e_tot)
        fc_tot += fc_this
        if not chg_scf_conv:
            logger.warn(mf, "  Charge SCF not converged.")

        # update coulomb potential, support vecs for PP & EXX, and energies
        mocc_ks = chg_mocc_ks
        vj_R = mf.get_vj_R(C_ks, mocc_ks)
        mf.update_pp(C_ks)
        mf.update_k(C_ks, mocc_ks)
        last_hf_moe = moe_ks
        moe_ks, mocc_ks = mf.get_mo_energy(C_ks, mocc_ks, vj_R=vj_R)
        # C_ks, moe_ks, mocc_ks = sort_mo(C_ks, moe_ks, mocc_ks)
        de_band = get_band_err(moe_ks, last_hf_moe, nband)
        last_hf_e = e_tot
        e_tot = mf.energy_tot(C_ks, mocc_ks, vj_R=vj_R)
        de = e_tot - last_hf_e

        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  max|dEband|= %4.3g  %d FC (%d tot)',
                    cycle+1, e_tot, de, de_band, fc_this, fc_tot)
        mf.dump_moe(moe_ks, mocc_ks, nband=nband)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(de) < conv_tol and abs(de_band) < conv_tol_band:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        cput1 = logger.timer(mf, 'cycle= %d'%(cycle+1), *cput1)

        if scf_conv:
            break

    if scf_conv and conv_check:
        # An extra diagonalization, to remove level shift
        chg_conv_tol = min(chg_conv_tol, max(conv_tol, 0.1*abs(de)))
        conv_tol_davidson = max(conv_tol*0.1, chg_conv_tol*0.01)
        logger.debug(mf, "  Performing charge SCF with conv_tol= %.3g conv_tol_davidson= %.3g", chg_conv_tol, conv_tol_davidson)

        chg_scf_conv, fc_this, C_ks, chg_moe_ks, chg_mocc_ks, chg_e_tot = \
                            mf.kernel_charge(
                                C_ks, mocc_ks, kpts, nband, mesh=mesh, Gv=Gv,
                                max_cycle=max_cycle, conv_tol=chg_conv_tol,
                                max_cycle_davidson=max_cycle_davidson,
                                conv_tol_davidson=conv_tol_davidson,
                                verbose_davidson=verbose_davidson,
                                damp_type=damp_type, damp_factor=damp_factor,
                                last_hf_e=e_tot)
        fc_tot += fc_this

        mocc_ks = chg_mocc_ks
        vj_R = mf.get_vj_R(C_ks, mocc_ks)
        mf.update_pp(C_ks)
        mf.update_k(C_ks, mocc_ks)
        last_hf_moe = moe_ks
        moe_ks, mocc_ks = mf.get_mo_energy(C_ks, mocc_ks, vj_R=vj_R)
        # C_ks, moe_ks, mocc_ks = sort_mo(C_ks, moe_ks, mocc_ks)
        de_band = get_band_err(moe_ks, last_hf_moe, nband)
        last_hf_e = e_tot
        e_tot = mf.energy_tot(C_ks, mocc_ks, vj_R=vj_R)
        de = e_tot - last_hf_e

        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  max|dEband|= %4.3g  %d FC (%d tot)',
                    e_tot, de, de_band, fc_this, fc_tot)
        mf.dump_moe(moe_ks, mocc_ks, nband=nband)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(de) < conv_tol and abs(de_band) < conv_tol_band:
            scf_conv = True

    # remove extra virtual bands before return
    remove_extra_virbands(C_ks, moe_ks, mocc_ks, nbandv_extra)

    if dump_chk:
        mf.dump_chk(locals())

    if callable(callback):
        callback(locals())

    if mf.outcore:
        C_ks = chkfile.load_mo_coeff(C_ks)
        fswap.close()

    cput1 = (time.clock(), time.time())
    mf.scf_summary["t-tot"] = np.asarray(cput1) - np.asarray(cput0)
    logger.debug(mf, '    CPU time for %s %9.2f sec, wall time %9.2f sec',
                 "scf_cycle", *mf.scf_summary["t-tot"])
    # A post-processing hook before return
    mf.post_kernel(locals())
    return scf_conv, e_tot, moe_ks, C_ks, mocc_ks


def get_nband(mf, nbandv, nbandv_extra):
    cell = mf.cell
    nbando = cell.nelectron // 2
    nbandv_tot = nbandv + nbandv_extra
    nband = nbando + nbandv
    nband_tot = nbando + nbandv_tot

    return nbando, nbandv_tot, nband, nband_tot


# def sort_mo(C_ks, moe_ks, mocc_ks, occ0=None):
#     if occ0 is None: occ0 = 2
#     if isinstance(moe_ks[0], np.ndarray):
#         nkpts = len(moe_ks)
#         for k in range(nkpts):
#             idxocc = np.where(mocc_ks[k]>THR_OCC)[0]
#             idxvir = np.where(mocc_ks[k]<THR_OCC)[0]
#             order = np.concatenate([idxocc, idxvir])
#             mocc_ks[k] = np.asarray([occ0 if i < len(idxocc) else 0
#                                     for i in range(len(order))])
#             moe_ks[k] = moe_ks[k][order]
#             set_kcomp(get_kcomp(C_ks, k)[order], C_ks, k)
#         return C_ks, moe_ks, mocc_ks
#     else:
#         ncomp = len(moe_ks)
#         for comp in range(ncomp):
#             C_ks_comp = get_kcomp(C_ks, comp, load=False)
#             C_ks_comp, moe_ks[comp], mocc_ks[comp] = sort_mo(C_ks_comp,
#                                                              moe_ks[comp],
#                                                              mocc_ks[comp],
#                                                              occ0=1)
#             if isinstance(C_ks, list): C_ks[comp] = C_ks_comp
#         return C_ks, moe_ks, mocc_ks


def get_band_err(moe_ks, last_hf_moe, nband):
    if isinstance(moe_ks[0], np.ndarray):
        if nband == 0: return 0.
        nkpts = len(moe_ks)
        return np.max([np.max(abs(moe_ks[k] - last_hf_moe[k])[:nband])
                      for k in range(nkpts)])
    else:
        ncomp = len(moe_ks)
        if isinstance(nband, int): nband = [nband] * ncomp
        if sum(nband) == 0: return 0.
        err = np.zeros(ncomp)
        for comp in range(ncomp):
            err[comp] = get_band_err(moe_ks[comp], last_hf_moe[comp],
                                     nband[comp])
        return np.max(err)


def remove_extra_virbands(C_ks, moe_ks, mocc_ks, nbandv_extra):
    if isinstance(moe_ks[0], np.ndarray):
        if nbandv_extra > 0:
            nkpts = len(moe_ks)
            for k in range(nkpts):
                n_k = len(moe_ks[k])
                occ = list(range(n_k-nbandv_extra))
                moe_ks[k] = moe_ks[k][occ]
                mocc_ks[k] = mocc_ks[k][occ]
                C = get_kcomp(C_ks, k, occ=occ)
                set_kcomp(C, C_ks, k)
    else:
        ncomp = len(moe_ks)
        if isinstance(nbandv_extra, int):
            nbandv_extra = [nbandv_extra] * ncomp
        for comp in range(ncomp):
            C_ks_comp = get_kcomp(C_ks, comp, load=False)
            remove_extra_virbands(C_ks_comp, moe_ks[comp], mocc_ks[comp],
                                  nbandv_extra[comp])


def kernel_charge(mf, C_ks, mocc_ks, kpts, nband, mesh=None, Gv=None,
                  max_cycle=50, conv_tol=1e-6,
                  max_cycle_davidson=10, conv_tol_davidson=1e-8,
                  verbose_davidson=0,
                  damp_type="anderson", damp_factor=0.3,
                  vj_R=None,
                  last_hf_e=None):

    cell = mf.cell
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
    if vj_R is None: vj_R = mf.get_vj_R(C_ks, mocc_ks, mesh=mesh, Gv=Gv)

    scf_conv = False

    fc_tot = 0

    if damp_type.lower() == "simple":
        chgmixer = pw_helper.SimpleMixing(mf, damp_factor)
    elif damp_type.lower() == "anderson":
        chgmixer = pw_helper.AndersonMixing(mf)

    cput1 = (time.clock(), time.time())
    for cycle in range(max_cycle):

        if cycle > 0:   # charge mixing
            vj_R = chgmixer.next_step(mf, vj_R, vj_R-last_vj_R)

        conv_ks, moe_ks, C_ks, fc_ks = mf.converge_band(
                            C_ks, mocc_ks, kpts,
                            mesh=mesh, Gv=Gv,
                            vj_R=vj_R,
                            conv_tol_davidson=conv_tol_davidson,
                            max_cycle_davidson=max_cycle_davidson,
                            verbose_davidson=verbose_davidson)
        fc_this = sum(fc_ks)
        fc_tot += fc_this

        # update mo occ
        mocc_ks = mf.get_mo_occ(moe_ks)

        # update coulomb potential and energy
        last_vj_R = vj_R
        vj_R = mf.get_vj_R(C_ks, mocc_ks)

        if cycle > 0: last_hf_e = e_tot
        e_tot = mf.energy_tot(C_ks, mocc_ks, vj_R=vj_R)
        if not last_hf_e is None:
            de = e_tot-last_hf_e
        else:
            de = float("inf")
        logger.debug(mf, '  chg cyc= %d E= %.15g  delta_E= %4.3g  %d FC (%d tot)',
                    cycle+1, e_tot, de, fc_this, fc_tot)
        mf.dump_moe(moe_ks, mocc_ks, nband=nband, trigger_level=logger.DEBUG3)

        if abs(de) < conv_tol:
            scf_conv = True

        cput1 = logger.timer_debug1(mf, 'chg cyc= %d'%(cycle+1),
                                    *cput1)

        if scf_conv:
            break

    return scf_conv, fc_tot, C_ks, moe_ks, mocc_ks, e_tot


def get_mo_occ(cell, moe_ks=None, C_ks=None, nocc=None):
    if nocc is None: nocc = cell.nelectron // 2
    if not moe_ks is None:
        nkpts = len(moe_ks)
        nocc_tot = nocc * nkpts
        e_fermi = np.sort(np.concatenate(moe_ks))[nocc_tot-1]
        EPSILON = 1e-10
        mocc_ks = [None] * nkpts
        for k in range(nkpts):
            mocc_k = np.zeros(moe_ks[k].size)
            mocc_k[moe_ks[k] < e_fermi+EPSILON] = 2
            mocc_ks[k] = mocc_k
    elif not C_ks is None:
        nkpts = len(C_ks)
        mocc_ks = [None] * nkpts
        for k in range(nkpts):
            C_k = get_kcomp(C_ks, k, load=False)
            mocc_ks[k] = np.asarray([2 if i < nocc else 0
                                     for i in range(C_k.shape[0])])
    else:
        raise RuntimeError

    return mocc_ks


def dump_moe(mf, moe_ks_, mocc_ks_, nband=None, trigger_level=logger.DEBUG):
    if mf.verbose >= trigger_level:
        kpts = mf.cell.get_scaled_kpts(mf.kpts)
        nkpts = len(kpts)
        if not nband is None:
            moe_ks = [moe_ks_[k][:nband] for k in range(nkpts)]
            mocc_ks = [mocc_ks_[k][:nband] for k in range(nkpts)]
        else:
            moe_ks = moe_ks_
            mocc_ks = mocc_ks_

        has_occ = np.where([(mocc_ks[k] > THR_OCC).any()
                           for k in range(nkpts)])[0]
        if len(has_occ) > 0:
            ehomo_ks = np.asarray([np.max(moe_ks[k][mocc_ks[k]>THR_OCC])
                                  for k in has_occ])
            ehomo = np.max(ehomo_ks)
            khomos = has_occ[np.where(abs(ehomo_ks-ehomo) < 1e-4)[0]]

            logger.info(mf, '  HOMO = %.15g  kpt'+' %d'*khomos.size,
                         ehomo, *khomos)

        has_vir = np.where([(mocc_ks[k] < THR_OCC).any()
                           for k in range(nkpts)])[0]
        if len(has_vir) > 0:
            elumo_ks = np.asarray([np.min(moe_ks[k][mocc_ks[k]<THR_OCC])
                                  for k in has_vir])
            elumo = np.min(elumo_ks)
            klumos = has_vir[np.where(abs(elumo_ks-elumo) < 1e-4)[0]]

            logger.info(mf, '  LUMO = %.15g  kpt'+' %d'*klumos.size,
                         elumo, *klumos)

        if len(has_occ) >0 and len(has_vir) > 0:
            logger.debug(mf, '  Egap = %.15g', elumo-ehomo)

        np.set_printoptions(threshold=len(moe_ks[0]))
        logger.debug(mf, '     k-point                  mo_energy')
        for k,kpt in enumerate(kpts):
            if mocc_ks is None:
                logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s',
                             k, kpt[0], kpt[1], kpt[2], moe_ks[k].real)
            else:
                logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s  %s',
                             k, kpt[0], kpt[1], kpt[2],
                             moe_ks[k][mocc_ks[k]>0].real,
                             moe_ks[k][mocc_ks[k]==0].real)
        np.set_printoptions(threshold=1000)


def orth_mo1(cell, C, mocc, thr_nonorth=1e-6, thr_lindep=1e-8, follow=True):
    """ orth occupieds and virtuals separately
    """
    orth = pw_helper.orth
    Co = C[mocc>THR_OCC]
    Cv = C[mocc<THR_OCC]
    # orth occ
    if Co.shape[0] > 0:
        Co = orth(cell, Co, thr_nonorth, thr_lindep, follow)
    # project out occ from vir and orth vir
    if Cv.shape[0] > 0:
        Cv -= lib.dot(lib.dot(Cv, Co.conj().T), Co)
        Cv = orth(cell, Cv, thr_nonorth, thr_lindep, follow)

    C = np.vstack([Co,Cv])

    return C


def orth_mo(cell, C_ks, mocc_ks, thr=1e-3):
    nkpts = len(mocc_ks)
    for k in range(nkpts):
        C_k = get_kcomp(C_ks, k)
        C_k = orth_mo1(cell, C_k, mocc_ks[k], thr)
        set_kcomp(C_k, C_ks, k)
        C_k = None

    return C_ks


def get_init_guess(cell0, kpts, basis=None, pseudo=None, nvir=0,
                   key="hcore", out=None):
    """
        Args:
            nvir (int):
                Number of virtual bands to be evaluated. Default is zero.
            out (h5py group):
                If provided, the orbitals are written to it.
    """

    if not out is None:
        assert(isinstance(out, h5py.Group))

    nkpts = len(kpts)

    if basis is None: basis = cell0.basis
    if pseudo is None: pseudo = cell0.pseudo
    cell = cell0.copy()
    cell.basis = basis
    if len(cell._ecp) > 0:  # use GTH to avoid the slow init time of ECP
        gth_pseudo = {}
        for iatm in range(cell0.natm):
            atm = cell0.atom_symbol(iatm)
            if atm in gth_pseudo:
                continue
            q = cell0.atom_charge(iatm)
            if q == 0:  # Ghost atom
                continue
            else:
                gth_pseudo[atm] = "gth-pade-q%d"%q
        logger.debug(cell0, "Using the GTH-PP for init guess: %s", gth_pseudo)
        cell.pseudo = gth_pseudo
        cell.ecp = cell._ecp = cell._ecpbas = None
    else:
        cell.pseudo = pseudo
    cell.ke_cutoff = cell0.ke_cutoff
    cell.verbose = 0
    cell.build()

    logger.info(cell0, "generating init guess using %s basis", cell.basis)

    if len(kpts) < 30:
        pmf = scf.KRHF(cell, kpts)
    else:
        pmf = scf.KRHF(cell, kpts).density_fit()

    if key.lower() == "cycle1":
        pmf.max_cycle = 0
        pmf.kernel()
        mo_coeff = pmf.mo_coeff
        mo_occ = pmf.mo_occ
    elif key.lower() in ["hcore", "h1e"]:
        h1e = pmf.get_hcore()
        s1e = pmf.get_ovlp()
        mo_energy, mo_coeff = pmf.eig(h1e, s1e)
        mo_occ = pmf.get_occ(mo_energy, mo_coeff)
    elif key.lower() == "scf":
        pmf.kernel()
        mo_coeff = pmf.mo_coeff
        mo_occ = pmf.mo_occ
    else:
        raise NotImplementedError("Init guess %s not implemented" % key)

    # TODO: support specifying nvir for each kpt (useful for e.g., metals)
    assert(isinstance(nvir, int) and nvir >= 0)
    nocc = cell0.nelectron // 2
    nmo_ks = [len(mo_occ[k]) for k in range(nkpts)]
    ntot = nocc + nvir
    ntot_ks = [min(ntot,nmo_ks[k]) for k in range(nkpts)]

    logger.debug1(cell0, "converting init MOs from GTO basis to PW basis")
    C_ks = pw_helper.get_C_ks_G(cell, kpts, mo_coeff, ntot_ks, out=out,
                                verbose=cell0.verbose)
    mocc_ks = [mo_occ[k][:ntot_ks[k]] for k in range(nkpts)]

    C_ks = orth_mo(cell0, C_ks, mocc_ks)

    C_ks, mocc_ks = add_random_mo(cell0, [ntot]*nkpts, C_ks, mocc_ks)

    return C_ks, mocc_ks


def add_random_mo(cell, n_ks, C_ks, mocc_ks):
    """ Add random MOs if C_ks[k].shape[0] < n_ks[k] for any k
    """
    nkpts = len(n_ks)
    for k in range(nkpts):
        n = n_ks[k]
        C0 = get_kcomp(C_ks, k)
        n0 = C0.shape[0]
        if n0 < n:
            n1 = n - n0
            logger.warn(cell, "Requesting more orbitals than currently have (%d > %d) for kpt %d. Adding %d random orbitals.", n, n0, k, n1)
            C = add_random_mo1(cell, n, C0)
            set_kcomp(C, C_ks, k)
            C = None

            mocc = mocc_ks[k]
            mocc_ks[k] = np.concatenate([mocc, np.zeros(n1,dtype=mocc.dtype)])
        C0 = None

    return C_ks, mocc_ks


def add_random_mo1(cell, n, C0):
    n0, ngrids = C0.shape
    if n == n0:
        return C0

    C1 = np.random.rand(n-n0, ngrids) + 0j
    C1 -= lib.dot(lib.dot(C1, C0.conj().T), C0)
    C1 = pw_helper.orth(cell, C1, 1e-3, follow=False)

    return np.vstack([C0,C1])


def init_guess_by_chkfile(cell, chkfile_name, nvir, project=None, out=None):
    from pyscf.pbc.scf import chkfile
    scf_dict = chkfile.load_scf(chkfile_name)[1]
    mocc_ks = scf_dict["mo_occ"]
    nkpts = len(mocc_ks)
    ntot_ks = [None] * nkpts
    for k in range(nkpts):
        nocc = np.sum(mocc_ks[k]>THR_OCC)
        ntot_ks[k] = max(nocc+nvir, len(mocc_ks[k]))

    if out is None: out = [None] * nkpts
    C_ks = out
    with h5py.File(chkfile_name, "r") as f:
        C0_ks = f["mo_coeff"]
        for k in range(nkpts):
            set_kcomp(get_kcomp(C0_ks, k), C_ks, k)

    C_ks, mocc_ks = init_guess_from_C0(cell, C_ks, ntot_ks, out=C_ks,
                                       mocc_ks=mocc_ks)

    return C_ks, mocc_ks


def init_guess_from_C0(cell, C0_ks, ntot_ks, out=None, mocc_ks=None):
    nkpts = len(C0_ks)
    if out is None: out = [None] * nkpts
    C_ks = out

    # discarded high-energy orbitals if chkfile has more than requested
    for k in range(nkpts):
        ntot = ntot_ks[k]
        C0_k = get_kcomp(C0_ks, k)
        if C0_k.shape[0] > ntot:
            C = C0_k[:ntot]
            if not mocc_ks is None:
                mocc_ks[k] = mocc_ks[k][:ntot]
        else:
            C = C0_k
        set_kcomp(C, C_ks, k)

    if mocc_ks is None:
        mocc_ks = get_mo_occ(cell, C_ks=C_ks)

    C_ks = orth_mo(cell, C_ks, mocc_ks)

    C_ks, mocc_ks = add_random_mo(cell, ntot_ks, C_ks, mocc_ks)

    return C_ks, mocc_ks


def update_pp(mf, C_ks):
    tick = np.asarray([time.clock(), time.time()])
    if not "t-ppnl" in mf.scf_summary:
        mf.scf_summary["t-ppnl"] = np.zeros(2)

    mf.with_pp.update_vppnloc_support_vec(C_ks)

    tock = np.asarray([time.clock(), time.time()])
    mf.scf_summary["t-ppnl"] += tock - tick


def update_k(mf, C_ks, mocc_ks):
    tick = np.asarray([time.clock(), time.time()])
    if not "t-ace" in mf.scf_summary:
        mf.scf_summary["t-ace"] = np.zeros(2)

    mf.with_jk.update_k_support_vec(C_ks, mocc_ks, mf.kpts)

    tock = np.asarray([time.clock(), time.time()])
    mf.scf_summary["t-ace"] += tock - tick


def eig_subspace(mf, C_ks, mocc_ks, mesh=None, Gv=None, vj_R=None, exxdiv=None,
                 comp=None):

    cell = mf.cell
    if vj_R is None: vj_R = mf.get_vj_R(C_ks, mocc_ks)
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
    if exxdiv is None: exxdiv = mf.exxdiv

    kpts = mf.kpts
    nkpts = len(kpts)
    moe_ks = [None] * nkpts
    for k in range(nkpts):
        kpt = kpts[k]
        C_k = get_kcomp(C_ks, k)
        Cbar_k = mf.apply_Fock_kpt(C_k, kpt, mocc_ks, mesh, Gv, vj_R, exxdiv,
                                   comp=comp, ret_E=False)
        F_k = lib.dot(C_k.conj(), Cbar_k.T)
        e, u = scipy.linalg.eigh(F_k)
        moe_ks[k] = e
        C_k = lib.dot(u.T, C_k)
        set_kcomp(C_k, C_ks, k)
        C_k = Cbar_k = None

    mocc_ks = get_mo_occ(cell, moe_ks=moe_ks)
    if exxdiv == "ewald":
        for k in range(nkpts):
            moe_ks[k][mocc_ks[k] > THR_OCC] -= mf._madelung

    return C_ks, moe_ks, mocc_ks


def apply_hcore_kpt(mf, C_k, kpt, mesh, Gv, with_pp, C_k_R=None, comp=None,
                    ret_E=False):
    r""" Apply hcore (kinetic and PP) opeartor to orbitals at given k-point.
    """
    es = np.zeros(3, dtype=np.complex128)

    tspans = np.zeros((3,2))
    tick = np.asarray([time.clock(), time.time()])

    tmp = pw_helper.apply_kin_kpt(C_k, kpt, mesh, Gv)
    Cbar_k = tmp
    es[0] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2
    tock = np.asarray([time.clock(), time.time()])
    tspans[0] = tock - tick

    if C_k_R is None: C_k_R = tools.ifft(C_k, mesh)
    tmp = with_pp.apply_vppl_kpt(C_k, mesh=mesh, C_k_R=C_k_R)
    Cbar_k += tmp
    es[1] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2
    tick = np.asarray([time.clock(), time.time()])
    tspans[1] = tick - tock

    tmp = with_pp.apply_vppnl_kpt(C_k, kpt, mesh=mesh, Gv=Gv, comp=comp)
    Cbar_k += tmp
    es[2] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2
    tock = np.asarray([time.clock(), time.time()])
    tspans[2] = tock - tick

    for ie_comp,e_comp in enumerate(mf.scf_summary["e_comp_name_lst"][:3]):
        key = "t-%s" % e_comp
        if not key in mf.scf_summary:
            mf.scf_summary[key] = np.zeros(2)
        mf.scf_summary[key] += tspans[ie_comp]

    if ret_E:
        if (np.abs(es.imag) > 1e-6).any():
            e_comp = mf.scf_summary["e_comp_name_lst"][:3]
            icomps = np.where(np.abs(es.imag) > 1e-6)[0]
            logger.warn(mf, "Energy has large imaginary part:" +
                     "%s : %s\n" * len(icomps),
                     *[s for i in icomps for s in [e_comp[i],es[i]]])
        es = es.real
        return Cbar_k, es
    else:
        return Cbar_k


def apply_jk_kpt(mf, C_k, kpt, mocc_ks, kpts, mesh, Gv, vj_R, with_jk,
                 exxdiv, C_k_R=None, comp=None, ret_E=False):
    r""" Apply non-local part of the Fock opeartor to orbitals at given k-point. The non-local part includes the exact exchange.
    """
    tspans = np.zeros((2,2))
    es = np.zeros(2, dtype=np.complex128)

    tick = np.asarray([time.clock(), time.time()])
    tmp = with_jk.apply_j_kpt(C_k, mesh=mesh, vj_R=vj_R, C_k_R=C_k_R)
    Cbar_k = tmp * 2.
    es[0] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2.
    tock = np.asarray([time.clock(), time.time()])
    tspans[0] = np.asarray(tock - tick).reshape(1,2)

    tmp = -with_jk.apply_k_kpt(C_k, kpt, mesh=mesh, Gv=Gv, exxdiv=exxdiv,
                               comp=comp)
    Cbar_k += tmp
    es[1] = np.einsum("ig,ig->", C_k.conj(), tmp)
    tick = np.asarray([time.clock(), time.time()])
    tspans[1] = np.asarray(tick - tock).reshape(1,2)

    for ie_comp,e_comp in enumerate(mf.scf_summary["e_comp_name_lst"][-2:]):
        key = "t-%s" % e_comp
        if not key in mf.scf_summary:
            mf.scf_summary[key] = np.zeros(2)
        mf.scf_summary[key] += tspans[ie_comp]

    if ret_E:
        if (np.abs(es.imag) > 1e-6).any():
            e_comp = mf.scf_summary["e_comp_name_lst"][-2:]
            icomps = np.where(np.abs(es.imag) > 1e-6)[0]
            logger.warn(mf, "Energy has large imaginary part:" +
                     "%s : %s\n" * len(icomps),
                     *[s for i in icomps for s in [e_comp[i],es[i]]])
        es = es.real
        return Cbar_k, es
    else:
        return Cbar_k


def apply_Fock_kpt(mf, C_k, kpt, mocc_ks, mesh, Gv, vj_R, exxdiv,
                   comp=None, ret_E=False):
    """ Apply Fock operator to orbitals at given k-point.
    """
    kpts = mf.kpts
    with_pp = mf.with_pp
    with_jk = mf.with_jk
    C_k_R = tools.ifft(C_k, mesh)
# 1e part
    res_1e = mf.apply_hcore_kpt(C_k, kpt, mesh, Gv, with_pp, comp=comp,
                                C_k_R=C_k_R, ret_E=ret_E)
# 2e part
    res_2e = mf.apply_jk_kpt(C_k, kpt, mocc_ks, kpts, mesh, Gv, vj_R, with_jk,
                             exxdiv, C_k_R=C_k_R, comp=comp, ret_E=ret_E)
    C_k_R = None

    if ret_E:
        Cbar_k = res_1e[0] + res_2e[0]
        es = np.concatenate([res_1e[1], res_2e[1]])
        return Cbar_k, es
    else:
        Cbar_k = res_1e + res_2e
        return Cbar_k


def get_mo_energy(mf, C_ks, mocc_ks, mesh=None, Gv=None, exxdiv=None,
                  vj_R=None, comp=None, ret_mocc=True):
    cell = mf.cell
    if vj_R is None: vj_R = mf.get_vj_R(C_ks, mocc_ks)
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
    if exxdiv is None: exxdiv = mf.exxdiv

    kpts = mf.kpts
    nkpts = len(kpts)
    moe_ks = [None] * nkpts
    for k in range(nkpts):
        kpt = kpts[k]
        C_k = get_kcomp(C_ks, k)
        Cbar_k = mf.apply_Fock_kpt(C_k, kpt, mocc_ks, mesh, Gv, vj_R,
                                   exxdiv, comp=comp, ret_E=False)
        moe_k = np.einsum("ig,ig->i", C_k.conj(), Cbar_k)
        C_k = Cbar_k = None
        if (moe_k.imag > 1e-6).any():
            logger.warn(mf, "MO energies have imaginary part %s for kpt %d",
                        moe_k, k)
        moe_ks[k] = moe_k.real

    mocc_ks = get_mo_occ(cell, moe_ks=moe_ks)
    if exxdiv == "ewald":
        for k in range(nkpts):
            moe_ks[k][mocc_ks[k] > THR_OCC] -= mf._madelung

    if ret_mocc:
        return moe_ks, mocc_ks
    else:
        return moe_ks


def energy_elec(mf, C_ks, mocc_ks, mesh=None, Gv=None, moe_ks=None,
                vj_R=None, exxdiv=None):
    ''' Compute the electronic energy
    Pass `moe_ks` to avoid the cost of applying the expensive vj and vk.
    '''
    cell = mf.cell
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
    if exxdiv is None: exxdiv = mf.exxdiv

    C_incore = isinstance(C_ks, list)

    kpts = mf.kpts
    nkpts = len(kpts)

    e_ks = np.zeros(nkpts)
    if moe_ks is None:
        if vj_R is None: vj_R = mf.get_vj_R(C_ks, mocc_ks)
        e_comp = np.zeros(5)
        for k in range(nkpts):
            kpt = kpts[k]
            occ = np.where(mocc_ks[k] > THR_OCC)[0]
            Co_k = get_kcomp(C_ks, k, occ=occ)
            e_comp_k = mf.apply_Fock_kpt(Co_k, kpt, mocc_ks, mesh, Gv,
                                         vj_R, exxdiv, ret_E=True)[1]
            e_ks[k] = np.sum(e_comp_k)
            e_comp += e_comp_k
        e_comp /= nkpts

        if exxdiv == "ewald":
            e_comp[mf.scf_summary["e_comp_name_lst"].index("ex")] += \
                                                        mf._etot_shift_ewald

        for comp,e in zip(mf.scf_summary["e_comp_name_lst"],e_comp):
            mf.scf_summary[comp] = e
    else:
        for k in range(nkpts):
            kpt = kpts[k]
            occ = np.where(mocc_ks[k] > THR_OCC)[0]
            Co_k = get_kcomp(C_ks, k, occ=occ)
            e1_comp = mf.apply_hcore_kpt(Co_k, kpt, mesh, Gv, mf.with_pp,
                                         ret_E=True)[1]
            e_ks[k] = np.sum(e1_comp) * 0.5 + np.sum(moe_ks[k][occ])
    e_scf = np.sum(e_ks) / nkpts

    if moe_ks is None and exxdiv == "ewald":
        # Note: ewald correction is not needed if e_tot is computed from moe_ks since the correction is already in the mo energy
        e_scf += mf._etot_shift_ewald

    return e_scf


def energy_tot(mf, C_ks, mocc_ks, moe_ks=None, mesh=None, Gv=None,
               vj_R=None, exxdiv=None):
    e_nuc = mf.scf_summary["nuc"]
    e_scf = mf.energy_elec(C_ks, mocc_ks, moe_ks=moe_ks, mesh=mesh, Gv=Gv,
                           vj_R=vj_R, exxdiv=exxdiv)
    e_tot = e_scf + e_nuc
    return e_tot


def converge_band_kpt(mf, C_k, kpt, mocc_ks, nband=None, mesh=None, Gv=None,
                      vj_R=None, comp=None,
                      conv_tol_davidson=1e-6,
                      max_cycle_davidson=100,
                      verbose_davidson=0):
    ''' Converge all occupied orbitals for a given k-point using davidson algorithm
    '''

    fc = [0]
    def FC(C_k_, ret_E=False):
        fc[0] += 1
        C_k_ = np.asarray(C_k_)
        Cbar_k_ = mf.apply_Fock_kpt(C_k_, kpt, mocc_ks, mesh, Gv,
                                    vj_R, "none",
                                    comp=comp, ret_E=False)
        return Cbar_k_

    tick = np.asarray([time.clock(), time.time()])

    kG = kpt + Gv if np.sum(np.abs(kpt)) > 1.E-9 else Gv
    dF = np.einsum("gj,gj->g", kG, kG) * 0.5
    precond = lambda dx, e, x0: dx/(dF - e)

    nroots = C_k.shape[0] if nband is None else nband

    conv, e, c = lib.davidson1(FC, C_k, precond,
                               nroots=nroots,
                               verbose=verbose_davidson,
                               tol=conv_tol_davidson,
                               max_cycle=max_cycle_davidson)
    c = np.asarray(c)

    tock = np.asarray([time.clock(), time.time()])
    key = "t-dvds"
    if not key in mf.scf_summary:
        mf.scf_summary[key] = np.zeros(2)
    mf.scf_summary[key] += tock - tick

    return conv, e, c, fc[0]


def converge_band(mf, C_ks, mocc_ks, kpts, Cout_ks=None,
                  mesh=None, Gv=None,
                  vj_R=None, comp=None,
                  conv_tol_davidson=1e-6,
                  max_cycle_davidson=100,
                  verbose_davidson=0):
    if vj_R is None: vj_R = mf.get_vj_R(C_ks, mocc_ks)

    nkpts = len(kpts)
    if Cout_ks is None: Cout_ks = C_ks
    conv_ks = [None] * nkpts
    moeout_ks = [None] * nkpts
    fc_ks = [None] * nkpts

    for k in range(nkpts):
        kpt = kpts[k]
        C_k = get_kcomp(C_ks, k)
        conv_, moeout_ks[k], Cout_k, fc_ks[k] = \
                    mf.converge_band_kpt(C_k, kpt, mocc_ks,
                                         mesh=mesh, Gv=Gv,
                                         vj_R=vj_R, comp=comp,
                                         conv_tol_davidson=conv_tol_davidson,
                                         max_cycle_davidson=max_cycle_davidson,
                                         verbose_davidson=verbose_davidson)
        set_kcomp(Cout_k, Cout_ks, k)
        conv_ks[k] = np.prod(conv_)

    return conv_ks, moeout_ks, Cout_ks, fc_ks


# class PWKRHF(lib.StreamObject):
class PWKRHF(mol_hf.SCF):
    '''PWKRHF base class. non-relativistic RHF using PW basis.
    '''

    outcore = getattr(__config__, 'pbc_pwscf_khf_PWKRHF_outcore', True)
    conv_tol = getattr(__config__, 'pbc_pwscf_khf_PWKRHF_conv_tol', 1e-6)
    conv_tol_davidson = getattr(__config__,
                                'pbc_pwscf_khf_PWKRHF_conv_tol_davidson', 1e-7)
    conv_tol_band = getattr(__config__, 'pbc_pwscf_khf_PWKRHF_conv_tol_band',
                            1e-4)
    max_cycle = getattr(__config__, 'pbc_pwscf_khf_PWKRHF_max_cycle', 100)
    max_cycle_davidson = getattr(__config__,
                                 'pbc_pwscf_khf_PWKRHF_max_cycle_davidson',
                                 100)
    verbose_davidson = getattr(__config__,
                               'pbc_pwscf_khf_PWKRHF_verbose_davidson', 0)
    ace_exx = getattr(__config__, 'pbc_pwscf_khf_PWKRHF_ace_exx', True)
    damp_type = getattr(__config__, 'pbc_pwscf_khf_PWKRHF_damp_type',
                        "anderson")
    damp_factor = getattr(__config__, 'pbc_pwscf_khf_PWKRHF_damp_factor', 0.3)
    conv_check = getattr(__config__, 'scf_hf_SCF_conv_check', True)
    check_convergence = None
    callback = None

    def __init__(self, cell, kpts=np.zeros((1,3)), ekincut=None,
        exxdiv=getattr(__config__, 'pbc_scf_PWKRHF_exxdiv', 'ewald')):

        if not cell._built:
            sys.stderr.write('Warning: cell.build() is not called in input\n')
            cell.build()

        self.cell = cell
        mol_hf.SCF.__init__(self, cell)

        self.kpts = kpts
        self.exxdiv = exxdiv
        if self.exxdiv == "ewald":
            self._madelung = tools.pbc.madelung(self.cell, self.kpts)
            self._etot_shift_ewald = -0.5*self._madelung*cell.nelectron
        self.scf_summary["nuc"] = self.cell.energy_nuc()
        self.scf_summary["e_comp_name_lst"] = ["kin", "ppl", "ppnl", "coul", "ex"]

        self.nvir = 0 # number of virtual bands to compute
        self.nvir_extra = 1 # to facilitate converging the highest virtual
        self.init_guess = "hcore"

        self.with_pp = None
        self.with_jk = None

        self._keys = self._keys.union(['cell', 'exxdiv'])

    def dump_flags(self):

        logger.info(self, '******** PBC PWSCF flags ********')
        logger.info(self, "ke_cutoff = %s", self.cell.ke_cutoff)
        logger.info(self, "mesh = %s (%d PWs)", self.cell.mesh,
                    np.prod(self.cell.mesh))
        logger.info(self, "SCF init guess = %s", self.init_guess)
        logger.info(self, "SCF conv_tol = %s", self.conv_tol)
        logger.info(self, "SCF max_cycle = %d", self.max_cycle)
        logger.info(self, "Num virtual bands to compute = %s", self.nvir)
        logger.info(self, "Num extra v-bands included to help convergence = %s",
                    self.nvir_extra)
        logger.info(self, "Band energy conv_tol = %s", self.conv_tol_band)
        logger.info(self, "Davidson conv_tol = %s", self.conv_tol_davidson)
        logger.info(self, "Davidson max_cycle = %d", self.max_cycle_davidson)
        logger.info(self, "Use ACE = %s", self.ace_exx)
        logger.info(self, "Damping method = %s", self.damp_type)
        if self.damp_type.lower() == "simple":
            logger.info(self, "Damping factor = %s", self.damp_factor)
        if self.chkfile:
            logger.info(self, 'chkfile to save SCF result = %s', self.chkfile)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])

        logger.info(self, 'kpts = %s', self.kpts)
        logger.info(self, 'Exchange divergence treatment (exxdiv) = %s',
                    self.exxdiv)

        cell = self.cell
        if ((cell.dimension >= 2 and cell.low_dim_ft_type != 'inf_vacuum') and
            isinstance(self.exxdiv, str) and self.exxdiv.lower() == 'ewald'):
            madelung = self._madelung
            logger.info(self, '    madelung (= occupied orbital energy shift) = %s', madelung)
            logger.info(self, '    Total energy shift due to Ewald probe charge'
                        ' = -1/2 * Nelec*madelung = %.12g',
                        madelung*cell.nelectron * -.5)

    def dump_scf_summary(self, verbose=logger.DEBUG):
        log = logger.new_logger(self, verbose)
        summary = self.scf_summary
        def write(fmt, key):
            if key in summary:
                log.info(fmt, summary[key])
        log.info('**** SCF Summaries ****')
        log.info('Total Energy =                    %24.15f', self.e_tot)
        write('Nuclear Repulsion Energy =        %24.15f', 'nuc')
        write('Kinetic Energy =                  %24.15f', 'kin')
        write('Local PP Energy =                 %24.15f', 'ppl')
        write('Non-local PP Energy =             %24.15f', 'ppnl')
        write('Two-electron Coulomb Energy =     %24.15f', 'coul')
        write('Two-electron Exchjange Energy =   %24.15f', 'ex')
        write('Empirical Dispersion Energy =     %24.15f', 'dispersion')
        write('PCM Polarization Energy =         %24.15f', 'epcm')
        write('EFP Energy =                      %24.15f', 'efp')
        if getattr(self, 'entropy', None):
            log.info('(Electronic) Entropy              %24.15f', self.entropy)
            log.info('(Electronic) Zero Point Energy    %24.15f', self.e_zero)
            log.info('Free Energy =                     %24.15f', self.e_free)

        def write_time(comp, t_comp, t_tot):
            tc, tw = t_comp
            tct, twt = t_tot
            rc = tc / tct * 100
            rw = tw / twt * 100
            log.info('CPU time for %10s %9.2f  ( %6.2f%% ), wall time %9.2f  ( %6.2f%% )', comp.ljust(10), tc, rc, tw, rw)

        t_tot = summary["t-tot"]
        write_time("init guess", summary["t-init"], t_tot)
        write_time("init ACE", summary["t-ace"], t_tot)
        t_fock = np.zeros(2)
        for op in summary["e_comp_name_lst"]:
            write_time("op %s"%op, summary["t-%s"%op], t_tot)
            t_fock += summary["t-%s"%op]
        t_dvds = np.clip(summary['t-dvds']-t_fock, 0, None)
        write_time("dvds other", t_dvds, t_tot)
        t_other = t_tot - summary["t-init"] - summary["t-ace"] - \
                    summary["t-dvds"]
        write_time("all other", t_other, t_tot)
        write_time("full SCF", t_tot, t_tot)

    def get_mo_occ(mf, moe_ks=None, C_ks=None, nocc=None):
        return get_mo_occ(mf.cell, moe_ks, C_ks, nocc)

    def get_init_guess(self, init_guess=None, nvir=None, chkfile=None, C0=None,
                       out=None):
        if init_guess is None: init_guess = self.init_guess
        if nvir is None: nvir = self.nvir
        if chkfile is None: chkfile = self.chkfile
        if init_guess[:3] == "chk" and C0 is None:
            C_ks, mocc_ks = self.init_guess_by_chkfile(chk=chkfile, nvir=nvir,
                                                       out=out)
            dump_chk = True
        else:
            # if isinstance(chkfile, str):
            #     fchk = h5py.File(chkfile, "w")
            #     dump_chk = True
            # else:
            #     # tempfile (discarded after SCF)
            #     swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            #     fchk = lib.H5TmpFile(swapfile.name)
            #     swapfile = None
            #     dump_chk = False

            # C_ks = fchk.create_group("mo_coeff")
            C_ks = out

            if C0 is None:
                C_ks, mocc_ks = self.get_init_guess_key(nvir=nvir,
                                                        key=init_guess,
                                                        out=C_ks)
            else:
                C_ks, mocc_ks = self.get_init_guess_C0(C0, nvir=nvir, out=C_ks)

        return C_ks, mocc_ks

    def get_init_guess_key(self, cell=None, kpts=None, basis=None, pseudo=None,
                           nvir=None, key="hcore", out=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if nvir is None: nvir = self.nvir

        if key in ["h1e","hcore","cycle1","scf"]:
            C_ks, mocc_ks = get_init_guess(cell, kpts,
                                           basis=basis, pseudo=pseudo,
                                           nvir=nvir, key=key, out=out)
        else:
            logger.warn(self, "Unknown init guess %s", key)
            raise RuntimeError

        return C_ks, mocc_ks

    def init_guess_by_chkfile(self, chk=None, nvir=None, project=None,
                              out=None):
        if chk is None: chk = self.chkfile
        if nvir is None: nvir = self.nvir
        return init_guess_by_chkfile(self.cell, chk, nvir, project=project,
                                     out=out)
    def from_chk(self, chk=None, project=None, kpts=None):
        return self.init_guess_by_chkfile(chk, project, kpts)

    def dump_chk(self, envs):
        if self.chkfile:
            chkfile.dump_scf(self.mol, self.chkfile,
                             envs['e_tot'], envs['moe_ks'],
                             envs['mocc_ks'], envs['C_ks'])
        return self

    def get_init_guess_C0(self, C0, nvir=None, out=None):
        if nvir is None: nvir = self.nvir
        nocc = self.cell.nelectron // 2
        ntot_ks = [nocc+nvir] * len(self.kpts)
        return init_guess_from_C0(self.cell, C0, ntot_ks, out=out)

    def get_vj_R(self, C_ks, mocc_ks, mesh=None, Gv=None):
        return self.with_jk.get_vj_R(C_ks, mocc_ks, mesh=mesh, Gv=Gv)

    def init_pp(self, with_pp=None, **kwargs):
        return pw_pseudo.pseudopotential(self, with_pp=with_pp, **kwargs)

    def init_jk(self, with_jk=None, ace_exx=None):
        if ace_exx is None: ace_exx = self.ace_exx
        return pw_jk.jk(self, with_jk=with_jk, ace_exx=ace_exx)

    def scf(self, C0=None, **kwargs):
        self.dump_flags()

        if self.with_pp is None:
            with_pp = getattr(kwargs, "with_pp", None)
            self.init_pp(with_pp=with_pp)

        if self.with_jk is None:
            with_jk = getattr(kwargs, "with_jk", None)
            self.init_jk(with_jk=with_jk)

        self.converged, self.e_tot, self.mo_energy, self.mo_coeff, \
                self.mo_occ = kernel_doubleloop(self, self.kpts,
                           C0=C0,
                           nbandv=self.nvir, nbandv_extra=self.nvir_extra,
                           conv_tol=self.conv_tol, max_cycle=self.max_cycle,
                           conv_tol_band=self.conv_tol_band,
                           conv_tol_davidson=self.conv_tol_davidson,
                           max_cycle_davidson=self.max_cycle_davidson,
                           verbose_davidson=self.verbose_davidson,
                           ace_exx=self.ace_exx,
                           damp_type=self.damp_type,
                           damp_factor=self.damp_factor,
                           conv_check=self.conv_check,
                           callback=self.callback,
                           **kwargs)
        self._finalize()
        return self.e_tot
    kernel = lib.alias(scf, alias_name='kernel')

    kernel_charge = kernel_charge
    get_nband = get_nband
    dump_moe = dump_moe
    update_pp = update_pp
    update_k = update_k
    eig_subspace = eig_subspace
    get_mo_energy = get_mo_energy
    apply_hcore_kpt = apply_hcore_kpt
    apply_jk_kpt = apply_jk_kpt
    apply_Fock_kpt = apply_Fock_kpt
    energy_elec = energy_elec
    energy_tot = energy_tot
    converge_band_kpt = converge_band_kpt
    converge_band = converge_band


if __name__ == "__main__":
    cell = gto.Cell(
        atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994",
        a = np.asarray(
            [[0.       , 1.78339987, 1.78339987],
            [1.78339987, 0.        , 1.78339987],
            [1.78339987, 1.78339987, 0.        ]]),
        basis="gth-szv",
        ke_cutoff=50,
        pseudo="gth-pade",
    )
    cell.build()
    cell.verbose = 6

    kmesh = [2,1,1]
    kpts = cell.make_kpts(kmesh)

    mf = PWKRHF(cell, kpts)
    mf.nvir = 4 # converge first 4 virtual bands
    mf.kernel()
    mf.dump_scf_summary()

    assert(abs(mf.e_tot - -10.673452914596) < 1.e-5)
