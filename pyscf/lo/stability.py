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
    mask = abs(g) > 1e-10
    x0[mask] = 1. / hdiag[mask]
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
    exponent = mlo.exponent

    tril_ijdx = numpy.tril_indices(mlo.norb, k=-1)
    tril_idx, tril_jdx = tril_ijdx
    thetapool = numpy.asarray([1,2,3])*0.25*numpy.pi

    def update_rotation_local_(u, theta, i, j):
        ui = u[:,i].copy()
        uj = u[:,j].copy()
        u[:,i] = ui*numpy.cos(theta) + uj*numpy.sin(theta)
        u[:,j] = -ui*numpy.sin(theta) + uj*numpy.cos(theta)

    u = mlo.identity_rotation()
    stable = True
    while True:
        Pij = mlo.atomic_pops(u).real
        Qi = lib.einsum('xii->xi', Pij)
        Qiexp = Qi**exponent
        Lij = (Qiexp[:,None,:] + Qiexp[:,:,None]).sum(axis=0)[tril_ijdx]
        dLij = numpy.zeros_like(Lij)
        thetas = numpy.zeros_like(Lij)

        for theta in thetapool:
            c = numpy.cos(theta)
            s = numpy.sin(theta)

            Qitild = (Qi*c**2)[:,:,None] + (Qi*s**2)[:,None,:] + 2*c*s*Pij
            Qjtild = (Qi*s**2)[:,:,None] + (Qi*c**2)[:,None,:] - 2*c*s*Pij
            dLijtild = (Qitild**exponent+Qjtild**exponent).sum(axis=0)[tril_ijdx] - Lij
            mask = dLijtild > dLij + mlo.conv_tol
            thetas[mask] = theta
            dLij[mask] = dLijtild[mask]

        idxs = numpy.where(dLij > mlo.conv_tol)[0]

        if idxs.size == 0:
            break

        # Remove overlapping pairs using a greedy algorithm
        stable = False
        done = numpy.zeros(mlo.norb, dtype=bool)
        for idx in idxs:
            i, j = tril_idx[idx], tril_jdx[idx]
            if done[i] or done[j]:
                continue
            done[i] = done[j] = True

            theta = thetas[idx]
            log.info('Rotating orbital pair (%d,%d) by %.2f Pi. delta_f= %.14g',
                      i, j, theta/numpy.pi, dLij[idx])
            e0 = mlo.cost_function(u)
            update_rotation_local_(u, theta, i, j)
            e1 = mlo.cost_function(u)
            print(f'{e0:.10f}  {e1:.10f}  {e1-e0:.10f}')

    if stable:
        log.info(f'{mlo.__class__.__name__} is stable in the Jacobi stability analysis')
        mo_coeff = mlo.mo_coeff
    else:
        mo_coeff = mlo.rotate_orb(u)

    if return_status:
        return mo_coeff, stable
    else:
        return mo_coeff
