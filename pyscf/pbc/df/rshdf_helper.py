import sys
import numpy as np
import scipy.special

from pyscf import gto as mol_gto
from pyscf import lib


def _normalize2s(es, q0=None):
    if q0 is None: q0 = (4*np.pi)**-0.5 # half_sph_norm
    norms = mol_gto.gaussian_int(2, es)

    return q0/norms

def _squarednormalize2s(es):
    norms = mol_gto.gaussian_int(2, es*2)

    return norms**-0.5

def _binary_search(func, xlo, xhi, f0, xprec, args=None, verbose=0):
    """ For a monotonically decreasing 'func', Search for smallest x s.t. func(x) < f0. The search is stopped when xhi - xlo < xprec, and return xhi.
    """
    log = lib.logger.Logger(sys.stdout, verbose)

    if args is None:
        f = lambda x: abs(func(x))
    else:
        f = lambda x: abs(func(x, *args))

    lo = f(xlo)
    hi = f(xhi)

    if lo < f0:
        return xlo, lo

    while hi > f0:
        xhi *= 1.5
        hi = f(xhi)

    while True:
        log.debug1("%.10f  %.10f  %.3e  %.3e" % (xlo, xhi, lo, hi))
        if xhi - xlo < xprec:
            return xhi, hi

        xmi = 0.5 * (xhi + xlo)
        mi = f(xmi)

        if mi > f0:
            lo = mi
            xlo = xmi
        else:
            hi = mi
            xhi = xmi

def _round2cell(R, dLs_uniq):
    if R > dLs_uniq[-2]:
        return dLs_uniq[-1]

    # idx_max = np.where(dLs_uniq <= R)[0][-1] + 1
    idx_max = np.searchsorted(dLs_uniq, R)
    Rnew = (dLs_uniq[idx_max] + dLs_uniq[idx_max+1]) * 0.5
    return Rnew

def _get_squared_dist(a,b):
    a2 = np.einsum("ij,ij->i", a, a)
    b2 = np.einsum("ij,ij->i", b, b)
    ab = a @ b.T
    d2 = a2[:,None] + b2 - 2. * ab
    d2[d2 < 0.] = 0.
    return d2

def _extra_prec_angular(es, ls, cs):
    """ extra_prec = c / c(1s with same exponent) * f**l
        with f = 0.3 if e < 1, and 1.5 otherwise (totally empirical).
    """
    from pyscf import gto as mol_gto
    extra_prec = 1.
    for e, l, c in zip(es, ls, cs):
        if l == 0:
            continue
        c1s = mol_gto.gaussian_int(2, e*2)**-0.5
        extra_prec *= c / c1s
        # purely empirical
        extra_prec *= 0.3**l if e < 1. else 1.5**l

    return extra_prec

def _estimate_Rc_R12_cut(eaux, ei, ej, omega, Ls, cell_vol, precision,
                         R0s=None):

    es = np.asarray([eaux,ei,ej])
    cs = _squarednormalize2s(es)

    # eaux, ei, ej = es
    # laux, li, lj = ls
    caux, ci, cj = cs

    if R0s is None:
        R0s = np.zeros((2,3))

    Ri, Rj = R0s

    # sort Ls
    dLs = np.linalg.norm(Ls,axis=1)
    idx = np.argsort(dLs)
    Ls_sort = Ls[idx]
    dLs_sort = dLs[idx]
    dLs_uniq = np.unique(dLs_sort.round(2))

    a0 = cell_vol ** 0.333333333
    # r0 = a0 * (4*np.pi**0.33333333)**-0.33333333

    q0 = caux * eaux**-1.5

    sumeij = ei + ej
    etaij2 = ei * ej / sumeij
    eta1 = (1./eaux + 1./sumeij) ** -0.5
    eta2 = (1./eaux + 1./sumeij + 1./omega**2.) ** -0.5

    fac = (np.pi*0.25)**3. * ci*cj * sumeij**-1.5 * q0
    dos = 24*np.pi / cell_vol * a0

    f0 = lambda R: np.abs(scipy.special.erfc(eta1*R) -
                          scipy.special.erfc(eta2*R)) / R

    # Determine the short-range fsum (fsr) and the effective per-cell state
    # number (rho0) respectively using explicit double lattice sum. For fsr,
    # the summation is taken within a sphere of radius 2*a0, beyond which
    # the continuum approximation for the density of states, i.e.,
    # 4pi/cell_vol * R^2, becomes a good one. For rho0, the summation is taken
    # within a sphere so that exp(-eta*R12**2.) decays to prec_sr at the
    # boundary.
    prec_sr = 1e-3
    R12_sr = (-np.log(prec_sr)/etaij2)**0.5
    Rc_sr = a0 * 2.1

    Ls_sr = Ls_sort[dLs_sort<max(R12_sr,Rc_sr)]
    Rcs_sr_raw = _get_squared_dist(ei*(Ls_sr+Ri),
                                  -ej*(Ls_sr+Rj))**0.5 / sumeij

    idx1, idx2 = np.where(Rcs_sr_raw < a0*1.1)
    R12s2_sr = np.linalg.norm(Ls_sr[idx1]+Ri - (Ls_sr[idx2]+Rj), axis=-1)**2.
    rho0 = np.sum(np.exp(-etaij2 * R12s2_sr))

    Rcs_sr = Rcs_sr_raw[Rcs_sr_raw < Rc_sr]
    mask_zero = Rcs_sr < 1.e-3
    v0 = 2*np.pi**-0.5*abs(eta1-eta2) * np.sum(mask_zero)
    fsr = np.sum(f0(Rcs_sr[~mask_zero])) + v0

    # estimate Rc_cut using its long-range behavior
    #     ~ fac * rho0 * dos * Rc**2. * f0(Rc) < precision
    f = lambda R: fac * rho0 * dos * R**2. * f0(R)

    xlo = a0
    xhi = np.max(dLs_uniq)
    Rc_cut = _binary_search(f, xlo, xhi, precision, 1.)[0]
    # Rc_cut = (np.ceil(Rc_cut / a0)+0.1) * a0
    Rc_cut = _round2cell(Rc_cut, dLs_uniq)

    # determine short-range R12_cut
    #     ~ fac * fsr * dos * R12**2. * exp(-etaij2 * R12**2.) < precision
    def _estimate_R12_cut(fac_, precision_):
        def _iter1(R_):
            tmp = precision_ / (fac_ * R12_**2.)
            if isinstance(tmp,np.ndarray):
                tmp[tmp > 1.] = 0.99
            elif tmp > 1.:
                tmp = 0.99

            return (-np.log(tmp) / etaij2)**0.5

        R12_ = 10   # init guess
        R12_ = _iter1(R12_)
        R12_ = _iter1(R12_)

        minR12 = a0*1.1 # include at least the nearest neighbors
        if isinstance(R12_,np.ndarray):
            R12_[R12_ < minR12] = minR12
        elif R12_ < minR12:
            R12_ = minR12

        return R12_

    fac_R12 = fac * fsr * dos
    R12_cut_sr = _estimate_R12_cut(fac_R12, precision)

    # determine long-range R12_cut
    #     ~ flr(Rc) * dos * R12**2 * exp(-etaij * R12**2.) < precision
    # where
    #     flr(Rc) ~ fac * dos * Rc**2. * f0(Rc)
    if Rc_cut > Rc_sr:
        Rcs_lr = np.arange(np.ceil(Rc_sr), np.ceil(Rc_cut)+0.1, 1.)
        flr = fac * dos * Rcs_lr**2. * f0(Rcs_lr)
        fac_R12 = flr * dos
        R12_cut_lst_lr = _estimate_R12_cut(fac_R12, precision)

        # combine sr and lr R12_cut
        Rcs_sr = np.arange(0,Rcs_lr[0]-0.9,1)
        Rc_loc = np.concatenate([Rcs_sr, Rcs_lr])
        R12_cut_lst = np.concatenate([[R12_cut_sr]*Rcs_sr.size, R12_cut_lst_lr])
    else:
        Rc_loc = np.arange(0,np.ceil(Rc_cut)+0.1,1)
        R12_cut_lst = np.ones(Rc_loc.size) * R12_cut_sr

    return Rc_loc, R12_cut_lst


def _estimate_Rc_R12_cut2(eaux, ei, ej, omega, Ls, cell_vol, precision,
                         R0s=None):

    es = np.asarray([eaux,ei,ej])
    cs = _squarednormalize2s(es)

    # eaux, ei, ej = es
    # laux, li, lj = ls
    caux, ci, cj = cs

    if R0s is None:
        R0s = np.zeros((2,3))

    Ri, Rj = R0s

    # sort Ls
    dLs = np.linalg.norm(Ls,axis=1)
    idx = np.argsort(dLs)
    Ls_sort = Ls[idx]
    dLs_sort = dLs[idx]
    dLs_uniq = np.unique(dLs_sort.round(2))

    a0 = cell_vol**0.333333333
    r0 = a0 * (0.75/np.pi)**0.33333333333

    q0 = caux * eaux**-1.5

    sumeij = ei + ej
    etaij2 = ei * ej / sumeij
    eta1 = (1./eaux + 1./sumeij) ** -0.5
    eta2 = (1./eaux + 1./sumeij + 1./omega**2.) ** -0.5

    fac = (np.pi*0.25)**3. * ci*cj * sumeij**-1.5 * q0
    dosc = 4*np.pi / cell_vol * a0
    dos12 = 12*np.pi / cell_vol * r0 # dos12*R12^2 = 4*pi/vol*((R12+r0)^3-R12^3)

    f0 = lambda R: np.abs(scipy.special.erfc(eta1*R) -
                          scipy.special.erfc(eta2*R)) / R

    # Determine the short-range fmax (fsrmax) and the effective per-cell state
    # number (rho0) respectively using explicit double lattice sum. For fsr,
    # the summation is taken within a sphere of radius 2*a0, beyond which
    # the continuum approximation for the density of states, i.e.,
    # 4pi/cell_vol * R^2, becomes a good one. For rho0, the summation is taken
    # within a sphere so that exp(-eta*R12**2.) decays to prec_sr at the
    # boundary.
    prec_sr = 1e-3
    R12_sr = (-np.log(prec_sr)/etaij2)**0.5
    Rc_sr = a0 * 2.1

    Ls_sr = Ls_sort[dLs_sort<max(R12_sr,Rc_sr)]
    Rcs_sr_raw = _get_squared_dist(ei*(Ls_sr+Ri),
                                  -ej*(Ls_sr+Rj))**0.5 / sumeij

    idx1, idx2 = np.where(Rcs_sr_raw < r0*1.1)
    R12s2_sr = np.linalg.norm(Ls_sr[idx1]+Ri - (Ls_sr[idx2]+Rj), axis=-1)**2.
    rho0 = np.sum(np.exp(-etaij2 * R12s2_sr))

    Rcs_sr = np.sort(np.unique(Rcs_sr_raw[Rcs_sr_raw < Rc_sr].round(1)))
    Rcs_sr[abs(Rcs_sr)<1.e-10] = dLs_sort[-1]   # effectively removing zero
    v0 = 2*np.pi**-0.5*abs(eta1-eta2)
    fsrmax = max(np.max(f0(Rcs_sr)), v0)

    # estimate Rc_cut using its long-range behavior
    #     ~ fac * rho0 * dosc * Rc**2. * f0(Rc) < precision
    f = lambda R: fac * rho0 * dosc * (R+0.5*a0)**2. * f0(R)

    xlo = a0
    xhi = np.max(dLs_uniq)
    Rc_cut = _binary_search(f, xlo, xhi, precision, 1.)[0]
    # Rc_cut = (np.ceil(Rc_cut / a0)+0.1) * a0
    Rc_cut = _round2cell(Rc_cut, dLs_uniq)

    # determine short-range R12_cut
    #     ~ fac * fsrmax * dos12 * R12**2. * exp(-etaij2 * R12**2.) < precision
    def _estimate_R12_cut(fac_, precision_):
        def _iter1(R_):
            tmp = precision_ / (fac_ * R12_**2.)
            if isinstance(tmp,np.ndarray):
                tmp[tmp > 1.] = 0.99
            elif tmp > 1.:
                tmp = 0.99

            return (-np.log(tmp) / etaij2)**0.5

        R12_ = 10   # init guess
        R12_ = _iter1(R12_)
        R12_ = _iter1(R12_)

        minR12 = a0*1.1 # include at least the nearest neighbors
        if isinstance(R12_,np.ndarray):
            R12_[R12_ < minR12] = minR12
        elif R12_ < minR12:
            R12_ = minR12

        return R12_

    fac_R12 = fac * fsrmax * dos12
    R12_cut_sr = _estimate_R12_cut(fac_R12, precision)

    # determine long-range R12_cut
    #     ~ flr(Rc) * dos12 * R12**2 * exp(-etaij * R12**2.) < precision
    # where
    #     flr(Rc) ~ fac * dosc * Rc**2. * f0(Rc)
    if Rc_cut > Rc_sr:
        Rcs_lr = np.arange(np.ceil(Rc_sr), np.ceil(Rc_cut)+0.1, 1.)
        flr = fac * dosc * Rcs_lr**2. * f0(Rcs_lr)
        fac_R12 = flr * dos12
        R12_cut_lst_lr = _estimate_R12_cut(fac_R12, precision)

        # combine sr and lr R12_cut
        Rcs_sr = np.arange(0,Rcs_lr[0]-0.9,1)
        Rc_loc = np.concatenate([Rcs_sr, Rcs_lr])
        R12_cut_lst = np.concatenate([[R12_cut_sr]*Rcs_sr.size, R12_cut_lst_lr])
    else:
        Rc_loc = np.arange(0,np.ceil(Rc_cut)+0.1,1)
        R12_cut_lst = np.ones(Rc_loc.size) * R12_cut_sr

    return Rc_loc, R12_cut_lst


def _estimate_Rc_R12_cut2_batch(cell, auxcell, omega, auxprecs):

    prec_sr = 1e-3
    ncell_sr = 2

    cell_vol = cell.vol
    a0 = cell_vol**0.333333333
    r0 = a0 * (0.75/np.pi)**0.33333333333

    # sort Ls
    Ls = cell.get_lattice_Ls()
    dLs = np.linalg.norm(Ls,axis=1)
    idx = np.argsort(dLs)
    Ls_sort = Ls[idx]
    dLs_sort = dLs[idx]
    dLs_uniq = np.unique(dLs_sort.round(2))

    natm = cell.natm
    nbas = cell.nbas
    bas_atom = np.asarray([cell.bas_atom(ib) for ib in range(nbas)])
    bas_by_atom = [np.where(bas_atom==iatm)[0] for iatm in range(natm)]

    auxnbas = auxcell.nbas
    auxbas_atom = np.asarray([auxcell.bas_atom(ib) for ib in range(auxnbas)])
    auxbas_by_atom = [np.where(auxbas_atom==iatm)[0] for iatm in range(natm)]

    es = np.asarray([np.min(cell.bas_exp(ib)) for ib in range(nbas)])
    auxes = np.asarray([np.min(auxcell.bas_exp(ibaux)) for ibaux in range(auxnbas)])

    cs = _squarednormalize2s(es)
    auxcs = _squarednormalize2s(auxes)

    q0s = auxcs * auxes**-1.5

    dosc = 4*np.pi / cell_vol * a0
    dos12 = 12*np.pi / cell_vol * r0 # dos12*R12^2 = 4*pi/vol*((R12+r0)^3-R12^3)

    def _estimate_R12_cut(fac_, precision_, minR12=None):
        def _iter1(R_):
            tmp = precision_ / (fac_ * R12_**2.)
            if isinstance(tmp,np.ndarray):
                tmp[tmp > 1.] = 0.99
            elif tmp > 1.:
                tmp = 0.99

            return (-np.log(tmp) / etaij2)**0.5

        R12_ = 10   # init guess
        R12_ = _iter1(R12_)
        R12_ = _iter1(R12_)

        if not minR12 is None:
            if isinstance(R12_,np.ndarray):
                R12_[R12_ < minR12] = minR12
            elif R12_ < minR12:
                R12_ = minR12

        return R12_

    atom_coords = cell.atom_coords()

    Rc_cut_mat = np.zeros([auxnbas,nbas,nbas])
    R12_cut_lst = []
    for Patm in range(cell.natm):
        Raux = atom_coords[Patm]
        eauxs = auxes[auxbas_by_atom[Patm]]
        cauxs = auxcs[auxbas_by_atom[Patm]]
        q0s = cauxs * eauxs**-1.5
        for iatm in range(cell.natm):
            Ri = atom_coords[iatm] - Raux
            for jatm in range(cell.natm):
                Rj = atom_coords[jatm] - Raux
                # TODO: fix me for large supercell
                # minR12 = np.linalg.norm(Ri - Rj) * 1.1
                minR12 = a0*1.1

                for ib in bas_by_atom[iatm]:
                    ei = es[ib]
                    ci = cs[ib]
                    for jb in bas_by_atom[jatm]:
                        if jb > ib: continue

                        ej = es[jb]
                        cj = cs[jb]

                        sumeij = ei + ej
                        etaij2 = ei*ej/sumeij

                        R12_sr = (-np.log(prec_sr)/etaij2)**0.5
                        Rc_sr = a0 * (ncell_sr + 0.1)
                        Ls_sr = Ls_sort[:np.searchsorted(
                                        dLs_sort,max(R12_sr,Rc_sr))]

                        Rcs_sr_raw = _get_squared_dist(ei*(Ls_sr+Ri),
                                                       -ej*(Ls_sr+Rj))**0.5 / sumeij

                        idx1, idx2 = np.where(Rcs_sr_raw < r0*1.1)
                        R12s2_sr = np.linalg.norm(Ls_sr[idx1]+Ri -
                                                  (Ls_sr[idx2]+Rj), axis=-1)**2.
                        rho0 = np.sum(np.exp(-etaij2 * R12s2_sr))

                        Rcs_sr = np.sort(np.unique(Rcs_sr_raw[
                                         Rcs_sr_raw < Rc_sr].round(1)))
                        if Rcs_sr[0] < 1.e-10:  # effectively removing zero
                            Rcs_sr[0] = dLs_sort[-1]

                        eta1s = 1./eauxs + 1./sumeij
                        eta2s = (eta1s + 1./omega**2.) ** -0.5
                        eta1s **= -0.5
                        v0s = 2*np.pi**-0.5*abs(eta1s-eta2s)
                        facs = (np.pi*0.25)**3. * ci*cj * sumeij**-1.5 * q0s

                        for ibaux_,ibaux in enumerate(auxbas_by_atom[Patm]):
                            precision = auxprecs[ibaux]

                            eaux = auxes[ibaux]
                            caux = auxcs[ibaux]

                            eta1 = eta1s[ibaux_]
                            eta2 = eta2s[ibaux_]
                            fac = facs[ibaux_]
                            v0 = v0s[ibaux_]

                            f0 = lambda R: np.abs(scipy.special.erfc(eta1*R) -
                                                  scipy.special.erfc(eta2*R)) / R

                            fsrmax = max(np.max(f0(Rcs_sr)), v0)

                            # estimate Rc_cut using its long-range behavior
                            #     ~ fac * rho0 * dosc * Rc**2. * f0(Rc) < precision
                            f = lambda R: fac * rho0 * dosc * (R+0.5*a0)**2. * f0(R)

                            xlo = a0
                            xhi = dLs_uniq[-1]
                            Rc_cut = _binary_search(f, xlo, xhi, precision, 1.)[0]
                            Rc_cut = _round2cell(Rc_cut, dLs_uniq)

                            # determine short-range R12_cut
                            #     ~ fac * fsrmax * dos12 * R12**2. * exp(-etaij2 * R12**2.) < precision
                            fac_R12 = fac * fsrmax * dos12
                            R12_cut_sr = _estimate_R12_cut(fac_R12, precision, minR12)

                            # determine long-range R12_cut
                            #     ~ flr(Rc) * dos12 * R12**2 * exp(-etaij * R12**2.) < precision
                            # where
                            #     flr(Rc) ~ fac * dosc * Rc**2. * f0(Rc)
                            nc_cut = int(np.ceil(Rc_cut))
                            nc_sr = int(np.ceil(Rc_sr))
                            if nc_cut > nc_sr:
                                Rcs_lr = np.arange(nc_sr, nc_cut+0.1, 1.)
                                flr = fac * dosc * Rcs_lr**2. * f0(Rcs_lr)
                                fac_R12 = flr * dos12
                                R12_cut_lst_lr = _estimate_R12_cut(fac_R12, precision, minR12)

                                # combine sr and lr R12_cut
                                R12_cut_lst_ = np.concatenate([[R12_cut_sr]*nc_sr,
                                               R12_cut_lst_lr])
                            else:
                                R12_cut_lst_ = np.ones(nc_cut+1) * R12_cut_sr
                            Rc_cut_mat[ibaux,ib,jb] = Rc_cut_mat[ibaux,jb,ib] = R12_cut_lst_.size-1
                            R12_cut_lst.append(R12_cut_lst_)

    nc_max = np.max(Rc_cut_mat).astype(int)+1
    R12_cut_mat = np.zeros([auxnbas,nbas,nbas,nc_max])
    ind = 0
    for Patm in range(natm):
        for iatm in range(natm):
            for jatm in range(natm):
                for ib in bas_by_atom[iatm]:
                    for jb in bas_by_atom[jatm]:
                        if jb > ib: continue
                        for ibaux in auxbas_by_atom[Patm]:
                            R12_cut_lst_ = R12_cut_lst[ind]
                            nc_ = R12_cut_lst_.size
                            R12_cut_mat[ibaux,ib,jb,:nc_] = R12_cut_lst_
                            R12_cut_mat[ibaux,ib,jb,nc_:] = R12_cut_lst_[-1]
                            R12_cut_mat[ibaux,jb,ib] = R12_cut_mat[ibaux,ib,jb]

                            ind += 1

    return Rc_cut_mat, R12_cut_mat
