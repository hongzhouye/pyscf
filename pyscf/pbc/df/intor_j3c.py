"""
1. dcuts
2. Rcuts
3. supmol <-- dcuts, Rcuts
4. shlpr data
"""

import ctypes
import numpy as np
from scipy.special import gamma, comb

from pyscf import gto as mol_gto
from pyscf.scf import _vhf
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf import lib
from pyscf.lib import logger
from pyscf.lib.parameters import BOHR
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique, KPT_DIFF_TOL
libpbc = lib.load_library('libpbc')

from pyscf.pbc.df.supmol import (get_refuniq_map, binary_search, get_norm,
                                 get_dist_mat)
from pyscf.pbc.df.intor_j2c import Gamma, get_multipole, make_supmol_j2c


def get_ovlp_dcut(bas_lst, precision, r0=None):
    """ Given a list of basis, determine cutoff radius for the ovlp between each unique shell pair to drop below "precision".

    Return:
        1d array of length nbas*(nbas+1)//2 with nbas=len(bas_lst).
    """
    mol = mol_gto.M(atom="H 0 0 0; H 0 0 0", basis=bas_lst)
    nbas = len(bas_lst)
    n2 = nbas*(nbas+1)//2

    def estimate1(ish,jsh,R0,R1):
        shls_slice = (ish,ish+1,nbas+jsh,nbas+jsh+1)
        prec = precision
        def fcheck(R):
            mol._env[mol._atm[1,mol_gto.PTR_COORD]] = R
            I = get_norm( mol.intor("int1e_ovlp", shls_slice=shls_slice) )
            return I < prec
        return binary_search(R0, R1, 1, True, fcheck)

    if r0 is None: r0 = 30
    R0 = r0 * 0.3
    R1 = r0
    dcuts = np.zeros(n2)
    ij = 0
    for i in range(nbas):
        for j in range(i+1):
            dcuts[ij] = estimate1(i,j,R0, R1)
            ij += 1
    return dcuts

def make_dijs_lst(dcuts, dstep):
    return [np.arange(0,dcut,dstep) for dcut in dcuts]

# __CGCORRLS = (2*np.arange(20)+1)**0.5
# def get_3c2e_Rcuts_for_d(mol, auxmol, ish, jsh, dij, omega, precision, fac_type,
#                          eta_correct=True, R_correct=True):
#     """ Determine for AO shlpr (ish,jsh) separated by dij, the cutoff radius for
#             2-norm( (ksh|v_SR(omega)|ish,jsh) ) < precision
#         If dij ~ 0., the exact integral is used for estimation; otherwise,
#
#             j3c(d,R) ~ (pi/8) * f(d)/(a1^l1*a2^l2*a12^1.5) *
#                         gamma_l3/a3^(l3+1.5) *
#                         Gamma(l3+1.5, eta*R^2) / R^(l3+1)
#
#         Similar to :func:`get_2c2e_Rcut`, the exact integral is multiplied by factor eta / R if "eta_correct" and "R_correct" are set to True.
#
#     Args:
#         mol/auxmol (Mole object):
#             Provide AO/aux basis info.
#         ish/jsh (int):
#             AO shl index.
#         dij (float):
#             Separation between ish and jsh; in BOHR
#         omega (float):
#             erfc(omega * r12) / r12
#         precision (float):
#             target precision.
#     """
#     nbasaux = auxmol.nbas
#     eks = [auxmol.bas_exp(ksh)[0] for ksh in range(nbasaux)]
#     lks = [int(auxmol.bas_angular(ksh)) for ksh in range(nbasaux)]
#     cks = [auxmol._libcint_ctr_coeff(ksh)[0,0] for ksh in range(nbasaux)]
#
#     if dij < 0.1:   # concentric
#         def get_bas_byshl(mol, i):
#             return [int(mol.bas_angular(i)),
#                     *np.vstack([mol.bas_exp(i), mol.bas_ctr_coeff(i).T]).T]
#         basi = get_bas_byshl(mol, ish)
#         basj = get_bas_byshl(mol, jsh)
#         ei = mol.bas_exp(ish).min()
#         ej = mol.bas_exp(jsh).min()
#         mol1 = mol_gto.M(atom="H1 0 0 0; H2 0 0 0; H3 0 0 0",
#                          basis={"H1": [basi], "H2": [basj], "H3": [[0,[1,1]]]},
#                          spin=None)
# # >>>>>>>> TODO: remove me after numerical tests
#         from pyscf import __config__
#         safe = getattr(__config__, "INTOR_J3C_SAFE", False)
# # <<<<<<<< END
#         def estimate1(ksh, R0,R1):
#             mol1._env[mol1._bas[2,mol_gto.PTR_EXP]] = eks[ksh]
#             mol1._bas[2,mol_gto.ANG_OF] = lks[ksh]
#             mol1._env[mol1._bas[2,mol_gto.PTR_COEFF]] = cks[ksh]
#             shls_slice = (0,1,1,2,2,3)
#             eta = 1/(1/(ei+ej)+1/eks[ksh]+1/omega**2.)
#             prec0 = precision * (min(eta,1.) if eta_correct else 1.)
#             def fcheck(R):
#                 prec = prec0 * (min(1./R,1.) if R_correct else 1.)
#                 mol1._env[mol1._atm[2,mol_gto.PTR_COORD]] = R
#                 # The SR formula may be inaccurate for high-l GTOs & big omega (e.g., omega >~ 0.5), but only for small R. Thus, using SR formula is okay for estimating Rcut.
#                 if safe:
#                     I = mol1.intor("int3c2e", shls_slice=shls_slice)
#                     with mol1.with_range_coulomb(abs(omega)):
#                         I -= mol1.intor("int3c2e", shls_slice=shls_slice)
#                 else:
#                     with mol1.with_range_coulomb(-abs(omega)):
#                         I = mol1.intor("int3c2e", shls_slice=shls_slice)
#                 I = get_norm( I )
#                 return I < prec
#             return binary_search(R0, R1, 1, True, fcheck)
#     else:
#         def get_cmpe_lst(n, d, e1, e2):
#             e12 = e1+e2
#             d1 = d * e2/e12
#             d2 = -d * e1/e12
#             fac0 = abs(d1**l1 * d2**l2)
#             cs_lst = [None] * (n+1)
#             cs_lst[0] = np.array([1.])
#             if n > 0:
#                 cs_lst[1] = np.array([
#                     l1/d1 if l1 > 0 else 0.,
#                     l2/d2 if l2 > 0 else 0.
#                 ])
#             if n > 1:
#                 cs_lst[2] = np.array([
#                     l1*(l1+1)*0.5/d1**2 if l1 > 1 else 0.,
#                     l2*(l2+1)*0.5/d2**2 if l2 > 1 else 0.,
#                     l1*l2/(d1*d2) if l1 > 0 and l2 > 0 else 0.
#                 ])
#             if n > 2:
#                 cs_lst[3] = np.array([
#                     l1*(l1+1)*(l1+2)*0.1667/d1**3 if l1 > 2 else 0.,
#                     l2*(l2+1)*(l2+2)*0.1667/d2**3 if l2 > 2 else 0.,
#                     l1*(l1+1)*l2*0.5/(d1**2*d2) if l1 > 1 and l2 > 0 else 0.,
#                     l2*(l2+1)*l1*0.5/(d2**2*d1) if l2 > 1 and l1 > 0 else 0.
#                 ])
#             if n > 3:
#                 cs_lst[4] = np.array([
#                     l1*(l1+1)*(l1+2)*(l1+3)*0.04167/d1**4 if l1 > 3 else 0.,
#                     l2*(l2+1)*(l2+2)*(l2+3)*0.04167/d2**4 if l2 > 3 else 0.,
#                     l1*(l1+1)*(l1+2)*l2*0.1667/(d1**3*d2) if l1>2 and l2>0 else 0.,
#                     l2*(l2+1)*(l2+2)*l1*0.1667/(d2**3*d1) if l2>2 and l1>0 else 0.,
#                     l1*(l1+1)*l2*(l2+1)*0.25/(d2**2*d2**2) if l1>1 and l2>1 else 0.
#                 ])
#             if n > 4:
#                 raise NotImplementedError
#             for i in range(n+1):
#                 cs_lst[i] *= fac0
#             return cs_lst
#         def estimate1_(e1,e2,e3,l1,l2,l3,c1,c2,c3, d, Rmin, Rmax, FAC_TYPE, Q):
#             e12 = e1 + e2
#             l12 = l1 + l2
#             eta12 = 1/(1/e1+1/e2)
#             eta1 = 1/(1/e12+1/e3)
#             eta2 = 1/(1/eta1+1/omega**2.)
#             common_fac = np.pi*0.125 * np.exp(-eta12*d**2.)/(e12**1.5) * \
#                         __CGCORRLS[l3]*e3**(-(l3+1.5)) * c1*c2*c3
#             if FAC_TYPE in ["ISF", "ISFQ", "MPE0", "MPE1", "MPE2", "MPE3",
#                             "MPE4"]:
#                 if FAC_TYPE == "ISF":
#                     fac = common_fac
#                 elif FAC_TYPE == "ISFQ":
#                     e12w = 1/(2/e12+1/omega**2.)
#                     Q2S = (0.5 * np.pi**0.5 / ((e12*0.5)**0.5 - e12w**0.5))**0.5
#                     S = Q * Q2S * 4*np.pi /(c1*c2)
#                     fac = common_fac * np.exp(eta12*d**2.) * S*(e12/np.pi)**1.5
#                 elif FAC_TYPE in ["MPE0","MPE1","MPE2","MPE3","MPE4"]:
#                     n = min(int(FAC_TYPE[-1]),l12)
#                     cs_lst = get_cmpe_lst(n, d, e1, e2)
#                     fac1 = sum([abs(cs_lst[i]).sum() for i in range(n+1)])
#                     fac = common_fac * fac1
#                 else:
#                     raise RuntimeError("Unknown fac type {}".format(fac_type))
#
#                 feval = lambda R: fac * Gamma(l3+0.5,eta2*R**2.) / R**(l3+1)
#
#             elif FAC_TYPE in ["MPE0L", "MPE1L", "MPE2L", "MPE3L", "MPE4L"]:
#                 n = min(int(FAC_TYPE[-2]),l12)
#                 cs_lst = get_cmpe_lst(n, d, e1, e2)
#
#                 def feval(R):
#                     I = 0.
#                     for m in range(n+1):
#                         fac = __CGCORRLS[m] / e12**m * abs(cs_lst[m]).sum()
#                         I += fac * Gamma(l3+m+0.5,eta2*R**2.) / R**(l3+m+1)
#                     I *= common_fac
#                     return I
#             else:
#                 raise RuntimeError("Unknown fac type {}".format(fac_type))
#
#             prec0 = precision * (min(eta2,1.) if eta_correct else 1.)
#             def fcheck(R):
#                 prec = prec0 * (min(1./R,1.) if R_correct else 1.)
#                 I = feval(R)
#                 return I < prec
#             return binary_search(Rmin, Rmax, 1, True, fcheck)
#
#         def get_lec(mol, i):
#             l = int(mol.bas_angular(i))
#             es = mol.bas_exp(i)
#             imin = es.argmin()
#             e = es[imin]
#             c = abs(mol._libcint_ctr_coeff(i)[imin]).max()
#             return l,e,c
#         l1,e1,c1 = get_lec(mol, ish)
#         l2,e2,c2 = get_lec(mol, jsh)
#
#         FAC_TYPE = fac_type.upper()
# # precompute Q
#         Q = None
#         if FAC_TYPE == "ISFQ":
#             mol12 = mol_gto.M(atom="H1 0 0 0; H2 %.10f 0 0" % (dij*BOHR),
#                               basis={"H1": [[l1,(e1,1.)]],
#                                      "H2": [[l2,(e2,1.)]]},
#                               spin=None)
#             with mol12.with_range_coulomb(-abs(omega)):
#                 Q = get_norm(
#                         mol12.intor("int2e", shls_slice=(0,1,1,2,0,1,1,2))
#                     )**0.5
# # precompute binary coefficients
#         def estimate1(ksh, R0,R1):
#             l3 = lks[ksh]
#             e3 = eks[ksh]
#             c3 = cks[ksh]
#             return estimate1_(e1,e2,e3,l1,l2,l3,c1,c2,c3, dij, R0, R1,
#                               FAC_TYPE, Q)
#
#     Rcuts = np.zeros(nbasaux)
#     R0 = 5
#     R1 = 20
#     for ksh in range(nbasaux):
#         Rcuts[ksh] = estimate1(ksh, R0, R1)
#
#     return Rcuts
def get_bincoeff(d,e1,e2,l1,l2):
    d1 = -e2/(e1+e2) * d
    d2 = e1/(e1+e2) * d
    lmax = l1+l2
    cbins = np.zeros(lmax+1)
    for l in range(0,lmax+1):
        cl = 0.
        lpmin = max(-l,l-2*l2)
        lpmax = min(l,2*l1-l)
        for lp in range(lpmin,lpmax+1,2):
            l1p = (l+lp) // 2
            l2p = (l-lp) // 2
            cl += d1**(l1-l1p)*d2**(l2-l2p) * comb(l1,l1p) * comb(l2,l2p)
        cbins[l] = cl
    return cbins
def get_3c2e_Rcuts_for_d(mol, auxmol, ish, jsh, dij, omega, precision, fac_type,
                         eta_correct=True, R_correct=True):
    """ Determine for AO shlpr (ish,jsh) separated by dij, the cutoff radius for
            2-norm( (ksh|v_SR(omega)|ish,jsh) ) < precision
        The estimator used here is
            ~ 0.5/pi * exp(-etaij*dij^2) * O_{k,lk} *
                \sum_{l=lmin}^{lmax} L_{li,lj}^{l} O_{ij,l} *
                Gamma(lk+l+1/2, eta2*R^2) / R^(lk+l+1)
        where
            eij = ei + ej
            lij = li + lj
            etaij = 1/(1/ei+1/ej)
            O_{k,lk} = 0.5*pi * (2*lk+1)^0.5 / ek^(lk+3/2)
            O_{ij,l} = 0.5*pi * (2*l+1)^0.5 / eij^(l+3/2)
            lmax = lij
            if d == 0:
                lmin = |li-lj|
                L_{li,lj}^{l} = eij^((l-lij)/2) * ((lij-1)!/(l-1)!)^0.5
            else:
                lmin = 0
                L_{li,lj}^{l} = \sum'_{m=-l}^{l} comb(li,mi) * comb(lj,mj) * di^(li-mi) * dj^(lj-mj)
                where
                    mi = (l+m)/2
                    mj = (l-m)/2
                    di = -ej/eij * (dij + extij)
                    dj = ei/eij * (dij + extij)
                where "extij" is the extent of orbital pair ij.

        Similar to :func:`get_2c2e_Rcut`, the estimator is multiplied by factor of eta and/or 1/R if "eta_correct" and/or "R_correct" are set to True.

    Args:
        mol/auxmol (Mole object):
            Provide AO/aux basis info.
        ish/jsh (int):
            AO shl index.
        dij (float):
            Separation between ish and jsh; in BOHR
        omega (float):
            erfc(omega * r12) / r12
        precision (float):
            target precision.
    """
# get bas info
    nbasaux = auxmol.nbas
    eks = [auxmol.bas_exp(ksh)[0] for ksh in range(nbasaux)]
    lks = [int(auxmol.bas_angular(ksh)) for ksh in range(nbasaux)]
    cks = [auxmol._libcint_ctr_coeff(ksh)[0,0] for ksh in range(nbasaux)]

    def get_lec(mol, i):
        l = int(mol.bas_angular(i))
        es = mol.bas_exp(i)
        imin = es.argmin()
        e = es[imin]
        c = abs(mol._libcint_ctr_coeff(i)[imin]).max()
        return l,e,c
    l1,e1,c1 = get_lec(mol, ish)
    l2,e2,c2 = get_lec(mol, jsh)

# local helper funcs
    def init_feval(e1,e2,e3,l1,l2,l3,c1,c2,c3, d):
        e12 = e1+e2
        l12 = l1+l2

        eta1 = 1/(1/e12+1/e3)
        eta2 = 1/(1/eta1+1/omega**2.)

        O3 = get_multipole(l3, e3)

        fac = c1*c2*c3 * O3 * 0.5/np.pi

        if d < 1e-3:    # concentric
            ls = np.arange(abs(l1-l2),l12+1)
            O12s = get_multipole(ls, e12)
            l_facs = O12s * e12**(0.5*(ls-l12)) * (
                            gamma(max(l12,1))/gamma(np.maximum(ls,1)))**0.5
        else:
            eta12 = 1/(1/e1+1/e2)
            fac *= np.exp(-eta12*d**2.)
            ls = np.arange(0,l12+1)
            O12s = get_multipole(ls, e12)
            l_facs = O12s * abs(get_bincoeff(d,e1,e2,l1,l2))

        def feval(R):
            I = 0.
            for l_fac,l in zip(l_facs,ls):
                I += l_fac * Gamma(l+l3+0.5,eta2*R**2.) / R**(l+l3+1)
            return I * fac

        return feval

    def estimate1(ksh, R0, R1):
        l3 = lks[ksh]
        e3 = eks[ksh]
        c3 = cks[ksh]
        feval = init_feval(e1,e2,e3,l1,l2,l3,c1,c2,c3, dij)

        eta2 = 1/(1/(e1+e2)+1/e2+1/omega**2.)
        prec0 = precision * (min(eta2,1.) if eta_correct else 1.)
        def fcheck(R):
            prec = prec0 * (min(1./R,1.) if R_correct else 1.)
            I = feval(R)
            return I < prec
        return binary_search(R0, R1, 1, True, fcheck)

# estimating Rcuts
    Rcuts = np.zeros(nbasaux)
    R0 = 5
    R1 = 20
    for ksh in range(nbasaux):
        Rcuts[ksh] = estimate1(ksh, R0, R1)

    return Rcuts
def get_3c2e_Rcuts(bas_lst, auxbas_lst, dijs_lst, omega, precision, fac_type,
                   eta_correct=True, R_correct=True):
    """ Given a list of basis ("bas_lst") and auxiliary basis ("auxbas_lst"), determine the cutoff radius for
        2-norm( (k|v_SR(omega)|ij) ) < precision
    where i and j shls are separated by d specified by "dijs_lst".
    """

    nbas = len(bas_lst)
    n2 = nbas*(nbas+1)//2
    nbasaux = len(auxbas_lst)

    mol = mol_gto.M(atom="H 0 0 0", basis=bas_lst, spin=None)
    auxmol = mol_gto.M(atom="H 0 0 0", basis=auxbas_lst, spin=None)

    ij = 0
    Rcuts = []
    for i in range(nbas):
        for j in range(i+1):
            dijs = dijs_lst[ij]
            for idij,dij in enumerate(dijs):
                Rcuts_dij = get_3c2e_Rcuts_for_d(mol, auxmol, i, j, dij,
                                                 omega, precision, fac_type,
                                                 eta_correct=eta_correct,
                                                 R_correct=R_correct)
                Rcuts.append(Rcuts_dij)
            ij += 1
    Rcuts = np.asarray(Rcuts).reshape(-1)
    return Rcuts

def get_atom_Rcuts(Rcuts, dijs_lst, bas_exps, bas_loc, auxbas_loc):
    natm = len(bas_loc) - 1
    assert(len(auxbas_loc) == natm+1)
    bas_loc_inv = np.concatenate([[i]*(bas_loc[i+1]-bas_loc[i])
                                  for i in range(natm)])
    nbas = bas_loc[-1]
    nbas2 = nbas*(nbas+1)//2
    nbasaux = auxbas_loc[-1]
    Rcuts_ = Rcuts.reshape(-1,nbasaux)
    dijs_loc = np.cumsum([0]+[len(dijs) for dijs in dijs_lst])
    betas = np.maximum(bas_exps[:,None],bas_exps) / (bas_exps[:,None]+bas_exps)

    atom_Rcuts = np.zeros((natm,natm))
    for katm in range(natm):    # aux atm
        k0, k1 = auxbas_loc[katm:katm+2]
        Rcuts_katm = np.max(Rcuts_[:,k0:k1], axis=1)

        rcuts_katm = np.zeros(natm)
        for ij in range(nbas2):
            i = int(np.floor((-1+(1+8*ij)**0.5)*0.5))
            j = ij - i*(i+1)//2
            ei = bas_exps[i]
            ej = bas_exps[j]
            bi = ej/(ei+ej)
            bj = ei/(ei+ej)
            dijs = dijs_lst[ij]
            idij0,idij1 = dijs_loc[ij:ij+2]
            rimax = (Rcuts_katm[idij0:idij1] + dijs*bi).max()
            rjmax = (Rcuts_katm[idij0:idij1] + dijs*bj).max()
            iatm = bas_loc_inv[i]
            jatm = bas_loc_inv[j]
            rcuts_katm[iatm] = max(rcuts_katm[iatm],rimax)
            rcuts_katm[jatm] = max(rcuts_katm[jatm],rjmax)

        atom_Rcuts[katm] = rcuts_katm

    return atom_Rcuts

def make_supmol_j3c(cell, atom_Rcuts, uniq_atms):
    return make_supmol_j2c(cell, atom_Rcuts, uniq_atms)

def get_shlpr_data(cell, supmol, dcuts, dijs_lst, refuniqshl_map,
                   dtype=np.int32):
    """
    Return:
        refshlprd_loc [size : nbas2+1]
        refshlprdinv_lst [size : nbas2d]
        supshlpr_loc [size : nbas2d+1]
        supshlpr_lst [size : nsupshlpr]
    """
    def tril_idx(i,j):
        return i*(i+1)//2+j if i>=j else j*(j+1)//2+i

    dijs_loc = np.cumsum([0]+[len(dijs) for dijs in dijs_lst])

    natm = cell.natm
    nbas = cell.nbas
    nbas2 = nbas*(nbas+1)//2
    refshl_loc = np.concatenate([cell.aoslice_nr_by_atom()[:,0],[nbas]])
    Rssup = supmol.atom_coords()
    supshlstart_by_atm = supmol.aoslice_nr_by_atom()[:,0]

    refshlprdinv_lst = [None] * nbas2
    supshlpr_loc = [None] * nbas2
    supshlpr_lst = [None] * nbas2 * 2
    for Iatm in range(natm):
        iatms = supmol._refsupatm_map[supmol._refsupatm_loc[Iatm]:
                                      supmol._refsupatm_loc[Iatm+1]]
        Ris = Rssup[iatms]
        for Jatm in range(Iatm+1):
            if Iatm == Jatm:
                jatms = iatms
                Rjs = Ris
            else:
                jatms = supmol._refsupatm_map[supmol._refsupatm_loc[Jatm]:
                                              supmol._refsupatm_loc[Jatm+1]]
                Rjs = Rssup[jatms]
            njatm = len(jatms)
            dijs = get_dist_mat(Ris, Rjs)
            dij_atmprids = np.arange(dijs.size)
            dijs = dijs.reshape(-1)

            Ish0, Ish1 = refshl_loc[Iatm:Iatm+2]
            for Ish in range(Ish0, Ish1):
                ISH = refuniqshl_map[Ish]
                if Iatm == Jatm:
                    Jsh0, Jsh1 = Ish0, Ish+1
                else:
                    Jsh0, Jsh1 = refshl_loc[Jatm:Jatm+2]
                for Jsh in range(Jsh0, Jsh1):
                    IJsh = tril_idx(Ish,Jsh)
                    JSH = refuniqshl_map[Jsh]
                    IJSH = tril_idx(ISH,JSH)

                    ids_keep = np.where(dijs <= dcuts[IJSH])[0]
                    if ids_keep.size == 0:
                        refshlprdinv_lst[IJsh] = np.array([], dtype=dtype)
                        supshlpr_loc[IJsh] = np.array([], dtype=dtype)
                        supshlpr_lst[IJsh] = np.array([], dtype=dtype)
                        supshlpr_lst[IJsh+nbas2] = np.array([], dtype=dtype)
                        continue

                    idij0 = dijs_loc[IJSH]
                    idij1 = dijs_loc[IJSH+1]
                    dijs_bins = dijs_lst[IJSH]
                    dijs_keep = dijs[ids_keep]
                    dijs_inverse = np.digitize(dijs_keep, bins=dijs_bins)-1
                    atmpr_keep_IJsh = []
                    dinv_IJsh = []
                    for idij,dij in enumerate(dijs_bins):
                        mask = dijs_inverse==idij
                        if np.any(mask):
                            atmpr_keep_IJsh.append(dij_atmprids[ids_keep[mask]])
                            dinv_IJsh.append(idij0+idij)

                    atmpr_len_IJsh = np.asarray([len(x)
                                                for x in atmpr_keep_IJsh],
                                                dtype=dtype)
                    atmpr_keep_IJsh = np.concatenate(atmpr_keep_IJsh)
                    iatms_keep = iatms[atmpr_keep_IJsh//njatm]
                    jatms_keep = jatms[atmpr_keep_IJsh%njatm]
                    ishs_keep = supshlstart_by_atm[iatms_keep] + (Ish-Ish0)
                    jshs_keep = supshlstart_by_atm[jatms_keep] + (Jsh-Jsh0)
                    # write
                    refshlprdinv_lst[IJsh] = np.asarray(dinv_IJsh, dtype=dtype)
                    supshlpr_loc[IJsh] = atmpr_len_IJsh
                    supshlpr_lst[IJsh] = ishs_keep.astype(dtype)
                    supshlpr_lst[IJsh+nbas2] = jshs_keep.astype(dtype)
                    # clear
                    atmpr_keep_IJsh = atmpr_len_IJsh = \
                        iatms_keep = jatms_keep = ishs_keep = jshs_keep = None

    refshlprd_loc = np.cumsum([0]+[len(x) for x in supshlpr_loc]).astype(dtype)
    refshlprdinv_lst = np.concatenate(refshlprdinv_lst)
    supshlpr_lst = np.concatenate(supshlpr_lst)
    supshlpr_loc = np.cumsum([0]+np.concatenate(
                             supshlpr_loc).tolist()).astype(dtype)

    return refshlprd_loc, refshlprdinv_lst, supshlpr_lst, supshlpr_loc

def intor_j3c(cell, auxcell, omega, kptijs=np.zeros((1,2,3)),
              precision=None, use_cintopt=True, safe=True, fac_type="MPE3",
# +++++++ Use the default for the following unless you know what you are doing
              eta_correct=True, R_correct=True,
              dstep=1,  # unit: Angstrom
# -------
# +++++++ debug options
              ret_timing=False,
              force_kcode=False,
# -------
              ):

    cput0 = np.asarray([logger.process_clock(), logger.perf_counter()])

    if precision is None: precision = cell.precision

    refuniqshl_map, uniq_atms, uniq_bas, uniq_bas_loc = get_refuniq_map(cell)
    auxuniqshl_map, uniq_atms, uniq_basaux, uniq_basaux_loc = \
                                                        get_refuniq_map(auxcell)
    nbasauxuniq = len(uniq_basaux)

    dcuts = get_ovlp_dcut(uniq_bas, precision, r0=cell.rcut)
    dijs_lst = make_dijs_lst(dcuts, dstep/BOHR)
    Rcuts = get_3c2e_Rcuts(uniq_bas, uniq_basaux, dijs_lst, omega, precision,
                           fac_type,
                           eta_correct=eta_correct, R_correct=R_correct)
    Rcut2s = Rcuts**2.
    bas_exps = np.array([np.asarray(b[1:])[:,0].min() for b in uniq_bas])
    atom_Rcuts = get_atom_Rcuts(Rcuts, dijs_lst, bas_exps, uniq_bas_loc,
                                uniq_basaux_loc)
    supmol = make_supmol_j3c(cell, atom_Rcuts, uniq_atms)
    refshlprd_loc, refshlprdinv_lst, supshlpr_lst, supshlpr_loc = \
                get_shlpr_data(cell, supmol, dcuts, dijs_lst, refuniqshl_map)
    refexp = np.asarray([cell.bas_exp(i).min() for i in range(cell.nbas)])

# concatenate atm/bas/env
    atm, bas, env = mol_gto.conc_env(cell._atm, cell._bas, cell._env,
                                     auxcell._atm, auxcell._bas, auxcell._env)
    atmsup, bassup, envsup = mol_gto.conc_env(
                                    supmol._atm, supmol._bas, supmol._env,
                                    auxcell._atm, auxcell._bas, auxcell._env)
    env[mol_gto.PTR_RANGE_OMEGA] = envsup[mol_gto.PTR_RANGE_OMEGA] = -abs(omega)

    dtype_idx = np.int32
    natm = cell.natm
    nbas = cell.nbas
    nbasaux = auxcell.nbas
    nbassup = supmol.nbas
    ao_loc = cell.ao_loc_nr()
    ao_locaux = auxcell.ao_loc_nr()
    ao_locsup = supmol.ao_loc_nr()
    ao_loc = np.concatenate([ao_loc[:-1], ao_locaux+ao_loc[-1]])
    ao_locsup = np.concatenate([ao_locsup[:-1], ao_locaux+ao_locsup[-1]])
    shl_loc = np.concatenate([cell.aoslice_nr_by_atom()[:,0],
                              auxcell.aoslice_nr_by_atom()[:,0]+nbas,
                              [nbas+nbasaux]]).astype(dtype_idx, copy=False)
    nsupshlpr = len(supshlpr_lst)//2
    nao = cell.nao
    naoaux = auxcell.nao

    nsupshlpr_tot = supmol.nbas*(supmol.nbas+1)//2
    logger.debug1(cell, "nsupshlpr_tot= %d  nsupshlpr_keep= %d  ( %.2f %% )",
                  nsupshlpr_tot, nsupshlpr, nsupshlpr/nsupshlpr_tot*100)
    memsp = nsupshlpr*8/1024**2.
    logger.debug1(cell, "mem use by shlpr data %.2f MB", memsp)

    intor = "int3c2e"
    intor, comp = mol_gto.moleintor._get_intor_and_comp(
                                            cell._add_suffix(intor), None)
    assert(comp == 1)
    if use_cintopt:
        cintopt = _vhf.make_cintopt(atmsup, bassup, envsup, intor)
    else:
        cintopt = lib.c_null_ptr()

    cput1 = np.asarray( logger.timer(cell, 'j3c precompute', *cput0) )
    dt0 = cput1 - cput0

    kptijs = np.asarray(kptijs).reshape(-1,2,3)

    if gamma_point(kptijs) and not force_kcode:
        drv = libpbc.fill_sr3c2e_g
        def fill_j3c(out):
            drv(getattr(libpbc, intor),
                out.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(comp), cintopt,
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                ao_locsup.ctypes.data_as(ctypes.c_void_p),
                shl_loc.ctypes.data_as(ctypes.c_void_p),
                auxuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbasauxuniq),
                Rcut2s.ctypes.data_as(ctypes.c_void_p),
                refexp.ctypes.data_as(ctypes.c_void_p),
                refshlprd_loc.ctypes.data_as(ctypes.c_void_p),
                refshlprdinv_lst.ctypes.data_as(ctypes.c_void_p),
                supshlpr_loc.ctypes.data_as(ctypes.c_void_p),
                supshlpr_lst.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nsupshlpr),
                atm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(natm),
                bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbas),
                ctypes.c_int(nbasaux),
                env.ctypes.data_as(ctypes.c_void_p),
                atmsup.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(supmol.natm),
                bassup.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(supmol.nbas),
                envsup.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_char(safe))

        nao2 = nao*(nao+1)//2
        memj3c = naoaux*nao2*8/1024**2.
        logger.debug1(cell, "estimated mem for 3c2e %.2f MB", memj3c)
        out = np.zeros((naoaux,nao2), dtype=np.double)
        with supmol.with_range_coulomb(-abs(omega)):
            fill_j3c(out)
        out = [out]
    else:
        nkptijs = len(kptijs)
        kpti = kptijs[:,0]
        kptj = kptijs[:,1]
        kpts = unique(np.vstack([kpti,kptj]))[0]
        expLk = np.exp(1j * lib.dot(supmol._Ls, kpts.T))
        wherei = np.where(abs(kpti[:,None,:]-kpts).sum(axis=2) <
                             KPT_DIFF_TOL)[1].astype(dtype_idx)
        wherej = np.where(abs(kptj[:,None,:]-kpts).sum(axis=2) <
                             KPT_DIFF_TOL)[1].astype(dtype_idx)
        nkpts = len(kpts)
        kptij_idx = np.concatenate([wherei,wherej])

        drv = libpbc.fill_sr3c2e_kk
        def fill_j3c(out):
            drv(getattr(libpbc, intor),
                out.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(comp), cintopt,
                expLk.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nkpts),
                kptij_idx.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nkptijs),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                ao_locsup.ctypes.data_as(ctypes.c_void_p),
                shl_loc.ctypes.data_as(ctypes.c_void_p),
                auxuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbasauxuniq),
                Rcut2s.ctypes.data_as(ctypes.c_void_p),
                refexp.ctypes.data_as(ctypes.c_void_p),
                refshlprd_loc.ctypes.data_as(ctypes.c_void_p),
                refshlprdinv_lst.ctypes.data_as(ctypes.c_void_p),
                supshlpr_loc.ctypes.data_as(ctypes.c_void_p),
                supshlpr_lst.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nsupshlpr),
                atm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(natm),
                bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbas),
                ctypes.c_int(nbasaux),
                env.ctypes.data_as(ctypes.c_void_p),
                atmsup.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(supmol.natm),
                bassup.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(supmol.nbas),
                envsup.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_char(safe))

        nao2 = nao*nao
        memj3c = nkptijs*naoaux*nao2*16/1024**2.
        logger.debug1(cell, "estimated mem for 3c2e %.2f MB", memj3c)
        out_ = np.zeros((nkptijs,naoaux,nao2), dtype=np.complex128)
        with supmol.with_range_coulomb(-abs(omega)):
            fill_j3c(out_)

        aosym_ks2 = abs(kpti-kptj).sum(axis=1) < KPT_DIFF_TOL
        tril_idx = np.tril_indices(nao)
        tril_idx = tril_idx[0] * nao + tril_idx[1]
        out = [None] * nkptijs
        for kij in range(nkptijs):
            v = out_[kij]
            if gamma_point(kptijs[kij]):
                v = v.real
            if aosym_ks2[kij]:
                v = v[:,tril_idx]
            out[kij] = v
        out_ = None

    cput0 = np.asarray( logger.timer(cell, 'j3c compute', *cput1) )
    dt1 = cput0 - cput1

    if ret_timing:
        return out, dt0, dt1
    else:
        return out


if __name__ == "__main__":
    from pyscf.pbc import gto, tools
    from pyscf import df
    from utils import get_lattice_sc40

    fml = "c"
    atom, a = get_lattice_sc40(fml)
    basis = "gth-dzvp"
    # basis = "cc-pvdz"
#     basis = """
# C    P
#       0.1517000              1.0000000
# C    D
#       0.5500000              1.0000000
# """
    # atom = "C 0 0 0"
    # a = np.eye(3) * 2.8
    # basis = [[0,[0.8,1.]], [1,[1,1.]]]

    cell = gto.Cell(atom=atom, a=a, basis=basis, spin=None)
    cell.verbose = 0
    cell.build()
    cell.verbose = 6

    # cell = tools.super_cell(cell, [2,2,2])

    auxcell = df.make_auxmol(cell)

    kptijs = np.zeros((1,2,3))
    # kptijs = np.asarray([np.random.rand(3)]*2).reshape(1,2,3)
    # kptijs = np.random.rand(1,2,3)
    nkpts = len(kptijs)

    omega = 0.8

    mesh = [31]*3
    from aft_j3c import j3c_aft
    # j3 = j3c_aft(cell, auxcell, omega, mesh, kptijs=kptijs)
    j3 = None

    prec_lst = [1e-8,1e-10]

    js = []
    j2s = []
    for prec in prec_lst:
        j, dt0, dt1 = intor_j3c(cell, auxcell, omega, kptijs=kptijs,
                                precision=prec)
        js.append(j)
        print("init time CPU %7.3f  wall %7.3f" % (dt0[0], dt0[1]))
        print("calc time CPU %7.3f  wall %7.3f" % (dt1[0], dt1[1]))

        if not j3 is None:
            for k in range(nkpts):

                # j[k] = lib.unpack_tril(j[k]).reshape(auxcell.nao,-1)

                err = abs(j[k] - j3[k])
                print("kpt %d "%k, err.max(), err.mean())
                err_r = abs(j[k].T.real-j3[k].T.real)
                print("real ", err_r.max(), err_r.mean())
                err_i = abs(j[k].T.imag-j3[k].T.imag)
                print("imag ", err_i.max(), err_i.mean())
                if j[k].shape[0] <= 10:
                    from frankenstein.tools.io_utils import dumpMat
                    # dumpMat(j[k].T.real)
                    # dumpMat(j3[k].T.real)
                    dumpMat(err_r, fmt="%.1e")
                    # dumpMat(j[k].T.imag*1e4)
                    # dumpMat(j3[k].T.imag*1e4)
                    dumpMat(err_i, fmt="%.1e")

        # from _3_intor import intor_j3c as intor_j3c_
        # j2, dt20, dt21 = intor_j3c_(cell, auxcell, omega, kpts=kpts, precision=prec,
        #               ret_timing=True)
        # j2s.append(j2)
        # print("init time CPU %7.3f  wall %7.3f" % (dt20[0], dt20[1]))
        # print("calc time CPU %7.3f  wall %7.3f" % (dt21[0], dt21[1]))
        #
        # if j[0].ndim == 2 and j[0].shape[1] <= 10:
        #     from frankenstein.tools.io_utils import dumpMat
        #     dumpMat(j[0])
        #     dumpMat(j2[0])
        # for k in range(nkpts):
        #     e = abs(j2[k]-j[k])
        #     print("kpt %d "%k, e.max(), e.mean())

    for k in range(nkpts):
        print("kpt %d" % k)
        for j in js[:-1]:
            errmat = abs(j[k]-js[-1][k])
            maxerr = errmat.max()
            meanerr = errmat.mean()
            print(" ", maxerr, meanerr)

        if len(j2s) > 0:
            for j2 in j2s[:-1]:
                errmat = abs(j2[k]-j2s[-1][k])
                maxerr = errmat.max()
                meanerr = errmat.mean()
                print(" ", maxerr, meanerr)
