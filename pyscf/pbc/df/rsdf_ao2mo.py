r''' AO2MO helper functions for DF 3c integrals
'''


import h5py
import numpy as np

from pyscf.ao2mo import _ao2mo
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf import lib
logger = lib.logger


def _ao2mo_e2(cderi, mo_coeffs, kpts, out=None):
    r''' (L|mu,nu) --> (L|i,j)

    Args:
        mo_coeffs (tuple or list):
            mo_coeffs = (C1_ks, C2_ks) where C1_ks and C2_ks are the mo coeff matrices
            for i and j for all kpts.
        out (numpy array of type "object" or h5py group):
            Where are the results stored?
            If numpy array, must be of type "object" and shape (nkpts,nkpts).
            If None, such a numpy array is created.
    '''
    from pyscf.pbc.df import df

    nkpts = len(kpts)

    mo_coeff1, mo_coeff2 = mo_coeffs
    assert(len(mo_coeff1) == nkpts and len(mo_coeff2) == nkpts)

    nao = mo_coeff1[0].shape[0]
    if gamma_point(kpts):
        dtype = np.double
    else:
        dtype = np.complex128
    dtype = np.result_type(dtype, *mo_coeff1, *mo_coeff2)

    if out is None:
        out = np.empty((nkpts, nkpts), dtype=object)
    elif isinstance(out, np.ndarray):
        assert(out.dtype == object)
    elif not isinstance(out, h5py.Group):
        raise TypeError('Input out must be np.ndarray or h5py.Group.')

    with h5py.File(cderi, 'r') as f:
        kptij_lst = f['j3c-kptij'][:]
        tao = []
        ao_loc = None
        for ki, kpti in enumerate(kpts):
            for kj, kptj in enumerate(kpts):
                kpti_kptj = np.array((kpti, kptj))
                Lpq_ao = np.asarray(df._getitem(f, 'j3c', kpti_kptj, kptij_lst))

                mo1 = mo_coeff1[ki]
                mo2 = mo_coeff2[kj]
                mo = np.asarray(np.hstack((mo1,mo2)), dtype=dtype, order='F')
                nmo1 = mo1.shape[1]
                nmo2 = mo2.shape[1]
                orbs_slice = (0, nmo1, nmo1, nmo1+nmo2)
                if dtype == np.double:
                    Lij = _ao2mo.nr_e2(Lpq_ao, mo, orbs_slice, aosym='s2')
                else:
                    #Note: Lpq.shape[0] != naux if linear dependency is found in auxbasis
                    if Lpq_ao[1].size != nao**2:  # aosym = 's2'
                        Lpq_ao = lib.unpack_tril(Lpq_ao).astype(np.complex128)
                    Lij = _ao2mo.r_e2(Lpq_ao, mo, orbs_slice, tao, ao_loc)
                if isinstance(out, np.ndarray):
                    out[ki, kj] = Lij.reshape(-1, nmo1, nmo2)
                else:
                    out['%d,%d'%(ki,kj)] = Lij.reshape(-1, nmo1, nmo2)

    return out
def ao2mo_e2(mydf, mo_coeffs, kpts=None, out=None):
    log = logger.new_logger(mydf)

    if mydf.cell.dimension < 3:
        raise NotImplementedError('DF ao2mo not implemented for low dimensions.')
    if kpts is None: kpts = mydf.kpts
    if mydf._cderi is None:
        mydf.build()
    cderi = mydf._cderi

    t0 = (logger.process_clock(), logger.perf_counter())
    out = _ao2mo_e2(cderi, mo_coeffs, kpts, out=out)
    log.timer('RSDF ao2mo', *t0)

    return out


if __name__ == '__main__':
    atom = 'He 0 0 0; He 1 0 0'
    a = np.eye(3) * 3
    basis = 'cc-pvdz'

    from pyscf.pbc import gto, scf
    cell = gto.Cell(atom=atom, a=a, basis=basis)
    cell.build()

    kmesh = [3,2,1]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.kernel()

    orbocc = [mo[:,occ>1e-10] for mo,occ in zip(mf.mo_coeff,mf.mo_occ)]
    orbvir = [mo[:,occ<=1e-10] for mo,occ in zip(mf.mo_coeff,mf.mo_occ)]
    mo_coeffs = (orbocc, orbvir)
    kkLov = ao2mo_e2(mf.with_df, mo_coeffs)
    kkLov_naive = ao2mo_e2_naive(mf.with_df, mo_coeffs)
    for k1 in range(nkpts):
        for k2 in range(nkpts):
            err = abs(kkLov[k1,k2] - kkLov_naive[k1,k2]).max()
            print(f'{k1:2d} {k2:2d}  {err:.3e}')
