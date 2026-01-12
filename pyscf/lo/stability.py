import numpy

from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__

from pyscf.scf.stability import STAB_NROOTS, STAB_TOL, dump_status


def stability_newton(mlo, verbose=None, return_status=False, nroots=STAB_NROOTS, tol=STAB_TOL):
    log = logger.new_logger(mlo, verbose)
    g, hop, hdiag = mlo.gen_g_hop()

    def precond(dx, e, x0):
        hdiagd = hdiag - e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return dx/hdiagd

    x0 = numpy.zeros_like(g)
    x0[g!=0] = 1. / hdiag[g!=0]
    x0 = numpy.vstack((x0, numpy.random.rand(5, x0.size)))  # add a few random vectors
    e, v = lib.davidson(hop, x0, precond, tol=tol, verbose=log.verbose-1, nroots=nroots)
    log.info('stability: lowest eigs of H = %s', e)
    if nroots != 1:
        e, v = e[0], v[0]
    stable = not (e < -1e-5)
    dump_status(log, stable, f'{mlo.__class__.__name__}', 'internal')
    if stable:
        mo = mlo.mo_coeff
    else:
        u = mlo.extract_rotation(v)
        mo = mlo.rotate_orb(u)
    if return_status:
        return mo, stable
    else:
        return mo


def stability_jacobi(mlo, verbose=None, return_status=False):
    ''' Check whether Jacobi sweep.
    '''
    log = logger.new_logger(mlo, verbose)

    triu_ijdx = numpy.triu_indices(mlo.norb, k=1)
    triu_idx, triu_jdx = triu_ijdx
    thetas = numpy.asarray([1,3,5,7])*0.25*numpy.pi
    c2ts = numpy.cos(thetas*2)
    s2ts = numpy.sin(thetas*2)

    def update_rotation_local_(u, theta, i, j):
        for x in u:
            xi = x[i].copy()
            xj = x[j].copy()
            x[i] = xi*numpy.cos(theta) - xj*numpy.sin(theta)
            x[j] = xi*numpy.sin(theta) + xj*numpy.cos(theta)

    u = mlo.identity_rotation()
    stable = True
    while True:
        proj = mlo.atomic_pops(u)

        # Lij = lib.einsum('xktij->xij', proj).real
        Lij = proj.real
        Lji = Lij.transpose(0,2,1)
        Lii = lib.einsum('xii->xi', Lij)
        Lijji = Lij + Lji
        Liijj = Lii[:,:,None] - Lii[:,None,:]
        L = numpy.triu( (Lijji**2 - Liijj**2).sum(axis=0), k=1 ) * 0.5

        Aij = (Lijji**2 - Liijj**2).sum(axis=0)[triu_ijdx]
        Bij = (Lijji * Liijj).sum(axis=0)[triu_ijdx]
        dLijt = Aij[:,None] * s2ts**2*0.5 - Bij[:,None] * s2ts*c2ts
        dLij = dLijt.max(axis=-1)

        idxs = numpy.where(dLij > mlo.conv_tol)[0]

        if idxs.size == 0:
            break

        # Remove overlapping pairs using a greedy algorithm
        stable = False
        done = numpy.zeros(mlo.norb, dtype=bool)
        for idx in idxs:
            i, j = triu_idx[idx], triu_jdx[idx]
            if done[i] or done[j]:
                continue
            done[i] = done[j] = True

            theta = thetas[dLijt[idx].argmax(axis=-1)]
            log.info('Rotating orbital pair (%d,%d) by %.2f Pi. delta_f= %.14g',
                      i, j, theta/numpy.pi, dLij[idx])
            update_rotation_local_(u, theta, i, j)

    if stable:
        mo_coeff = mlo.mo_coeff
    else:
        mo_coeff = mlo.rotate_orb(u)

    if return_status:
        return mo_coeff, stable
    else:
        return mo_coeff
