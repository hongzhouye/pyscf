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

    idx_max = np.where(dLs_uniq <= R)[0][-1] + 1
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
        R12_ = 10
        R12_ = (-np.log(precision_ / (fac_ * R12_**2.)) / etaij2)**0.5
        R12_ = (-np.log(precision_ / (fac_ * R12_**2.)) / etaij2)**0.5
        minR12 = a0 * 1.1
        if isinstance(R12_, float):
            if np.isnan(R12_):
                R12_ = minR12
            elif R12_ < minR12:
                R12_ = minR12
        else:
            R12_[np.isnan(R12_)] = minR12
            R12_[R12_ < minR12] = minR12

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
