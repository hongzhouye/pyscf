r''' AO2MO helper functions for direct DF 3c integrals
'''


import h5py
import numpy as np
import scipy.linalg

from pyscf.ao2mo import _ao2mo
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf import lib
from pyscf.pbc.df.rsdf_direct_helper import (loop_j3c, get_kptij_lst, loop_uniq_q,
                                             get_j2c, cholesky_decomposed_metric)
from pyscf.pbc.df.rsdf_direct_jk import _safe_member
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.lib import logger

from pyscf import __config__
AO2MO_KERNEL = getattr(__config__, 'pbc_gto_df_rsdf_ao2mo_direct_ao2mo_kernel', 1)


def ao2mo_e2_Lij_kernel1(mydf, mo_coeffs, kpts, bvk_kmesh=None, out=None):
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
    log = logger.new_logger(mydf)
    verbose1 = mydf.verbose - 2

    nkpts = len(kpts)

    mo_coeff1, mo_coeff2 = mo_coeffs
    assert(len(mo_coeff1) == nkpts and len(mo_coeff2) == nkpts)

    nao = mo_coeff1[0].shape[0]
    naoaux = mydf.auxcell.nao_nr()
    if gamma_point(kpts):
        dtype = np.double
        dsize = 8
    else:
        dtype = np.complex128
        dsize = 16
    dtype = np.result_type(dtype, *mo_coeff1, *mo_coeff2)

    if out is None:
        out = np.empty((nkpts,nkpts), dtype=object)
    elif isinstance(out, np.ndarray):
        assert(out.shape == (nkpts,nkpts))
        assert(out.dtype == object)
    elif not isinstance(out, h5py.Group):
        raise TypeError('Input out must be np.ndarray or h5py.Group.')

    incore = isinstance(out, np.ndarray)
    if incore:
        hasdata = _hasdata_incore
        loaddata = _loaddata_incore
        writedata = _writedata_incore
        accumdata = _accumdata_incore
        log.debug1('transformed integrals will be held incore.')
    else:
        hasdata = _hasdata_outcore
        loaddata = _loaddata_outcore
        writedata = _writedata_outcore
        accumdata = _accumdata_outcore
        log.debug1('transformed integrals will be saved to specified h5py file.')

    kptij_lst = get_kptij_lst(kpts)
    nkptij = len(kptij_lst)
    uniq_q_loop = [x for x in loop_uniq_q(mydf, kptij_lst=kptij_lst, verbose=0)]
    uniq_kpts = [x[0] for x in uniq_q_loop]
    nkpts_uniq = len(uniq_kpts)
    nkptjmax = np.max([len(x[1]) for x in uniq_q_loop])
    nkptijswap = sum([1 for x in uniq_q_loop for kptj in x[1]
                      if _safe_member(kptj, kpts)!=_safe_member(kptj-x[0], kpts)])

# evaluate and invert j2c
    t0 = (logger.process_clock(), logger.perf_counter())
    kj2c = get_j2c(mydf, kpts=uniq_kpts, verbose=verbose1)
    kj2c_negative = [None] * nkpts_uniq
    kj2ctag = [None] * nkpts_uniq
    for k,kpt in enumerate(uniq_kpts):
        kj2c[k], kj2c_negative[k], kj2ctag[k] = cholesky_decomposed_metric(mydf, kj2c[k])
    t0 = log.timer_debug1('ao2mo j2c', *t0)

# buffer size
    nmo1s = np.array([mo.shape[1] for mo in mo_coeff1])
    nmo2s = np.array([mo.shape[1] for mo in mo_coeff2])
    size_Lij0 = int(lib.einsum('i,j->ij',nmo1s,nmo2s).sum())
    size_Lij = 2    # "intermediate" Lij
    if incore: size_Lij += nkpts**2
    size_Lij *= size_Lij0
    mem_Lij = size_Lij * dsize / 1e6

    mem_avail = mydf.max_memory - lib.current_memory()[0] - mem_Lij
    size_Lpqblk = (nkptij+nkptjmax+1)*naoaux    # add 1 for potential mem use in einsum
    mem_Lpqblk = size_Lpqblk * dsize / 1e6
    aopblksize = min(nao*nao, int(np.floor(mem_avail*0.7/mem_Lpqblk)))
    shranges = _guess_shell_ranges(mydf.cell, aopblksize, 's1')
    aopblksize = np.max([x[2] for x in shranges])
    pblksize = aopblksize // nao
    log.debug1('ao2mo mem_avail= %.2f MB  mem_Lij= %.2f MB  mem_Lpqblk= %.2f MB',
               mem_avail, mem_Lij, mem_Lpqblk)
    log.debug1('ao2mo aopblksize= %d  pblksize= %d  nblk= %d', aopblksize, pblksize,
               len(shranges))
    log.debug1('ao2mo shranges= %s', shranges)

    tspans = np.zeros((6,2))
    tnames = ['ki,kj xform', 'ki,kj write', 'kj,ki xform', 'kj,ki write', 'xform', 'j3c']

    t1_tock = logger.process_clock(), logger.perf_counter()

    p1 = 0
    for kcLpq in loop_j3c(mydf, kptij_lst=kptij_lst, aosym='s1', partition_iorj='i',
                          j3c_order='Lij', shranges=shranges, bvk_kmesh=bvk_kmesh,
                          verbose=verbose1):
        dp = kcLpq.shape[-1] // nao
        assert(dp*nao == kcLpq.shape[-1])
        p0 = p1
        p1 += dp

        for kpt,adapted_kptjs,adapted_ji_idx in uniq_q_loop:
            for kptj,ji in zip(adapted_kptjs,adapted_ji_idx):
                kj = _safe_member(kptj, kpts)
                ki = _safe_member(kptj-kpt, kpts)

                tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                Lpq = kcLpq[ji][0].reshape(naoaux,dp,nao)
                mo1 = mo_coeff1[ki][p0:p1]
                mo2 = mo_coeff2[kj]
                nmo1 = mo1.shape[1]
                nmo2 = mo2.shape[1]
                Lij = lib.einsum('Lpq,pi,qj->Lij', Lpq, mo1.conj(), mo2)
                tock = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[0] += tock - tick
                Lij = Lij.reshape(-1,nmo1,nmo2)
                if hasdata(out,ki,kj):
                    accumdata(out,ki,kj,Lij)
                else:
                    writedata(out,ki,kj,Lij)
                Lij = None
                tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[1] += tick - tock

                if ki != kj:
                    mo1 = mo_coeff1[kj]
                    mo2 = mo_coeff2[ki][p0:p1]
                    nmo1 = mo1.shape[1]
                    nmo2 = mo2.shape[1]
                    Lji = lib.einsum('Lpq,pi,qj->Lij', Lpq, mo2.conj(), mo1).conj()
                    tock = np.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[2] += tock - tick
                    Lij = Lji.reshape(-1,nmo2,nmo1).transpose(0,2,1)
                    if hasdata(out,kj,ki):
                        accumdata(out,kj,ki,Lij)
                    else:
                        writedata(out,kj,ki,Lij)
                    tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[3] += tick - tock
                Lpq = Lij = Lji = None

        t1_tick = t1_tock
        t1_tock = log.timer_debug1('ao2mo pass1 [%d:%d]'%(p0,p1), *t1_tick)
        tspans[5] += np.asarray(t1_tock) - np.asarray(t1_tick)

    tspans[4] = tspans[:4].sum(axis=0)
    tspans[5] -= tspans[4]
    for tspan,tname in zip(tspans,tnames):
        log.debug1('CPU time for ao2mo pass1     %12s  %9.2f sec, '
                   'wall time  %9.2f sec', tname, *tspan)
    for tspan,tname in zip(tspans,tnames):
        if 'ki,kj' in tname or 'kj,ki' in tname:
            tspan_avg = tspan / max(1, nkptij if 'ji' in tname else nkptijswap)
            log.debug1('CPU time for ao2mo pass1 avg %12s  %9.2f sec, '
                       'wall time  %9.2f sec', tname, *tspan_avg)
    t0 = log.timer_debug1('ao2mo pass1', *t0)

    tspans = np.zeros((6,2))
    tnames = ['ki,kj  load','ki,kj solve','ki,kj write',
              'kj,ki  load','kj,ki solve','kj,ki write']

    kq = 0
    for kpt,adapted_kptjs,adapted_ji_idx in uniq_q_loop:
        j2c = kj2c[kq]
        j2ctag = kj2ctag[kq]
        for kptj,ji in zip(adapted_kptjs,adapted_ji_idx):
            tick = np.asarray((logger.process_clock(), logger.perf_counter()))
            kj = _safe_member(kptj, kpts)
            ki = _safe_member(kptj-kpt, kpts)
            Lij = loaddata(out,ki,kj)
            tock = np.asarray((logger.process_clock(), logger.perf_counter()))
            tspans[0] += tock - tick
            Lij = scipy.linalg.solve_triangular(j2c, Lij, lower=True)
            tick = np.asarray((logger.process_clock(), logger.perf_counter()))
            tspans[1] += tick - tock
            writedata(out,ki,kj,Lij)
            tock = np.asarray((logger.process_clock(), logger.perf_counter()))
            tspans[2] += tock - tick
            Lij = None
            if ki != kj:
                Lij = loaddata(out,kj,ki)
                tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[3] += tick - tock
                Lij = scipy.linalg.solve_triangular(j2c.conj(), Lij, lower=True)
                tock = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[4] += tock - tick
                writedata(out,kj,ki,Lij)
                tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[5] += tick - tock
                Lij = None
        kq += 1

    for tspan,tname in zip(tspans,tnames):
        log.debug1('CPU time for ao2mo pass2     %12s  %9.2f sec, '
                   'wall time  %9.2f sec', tname, *tspan)
    for tspan,tname in zip(tspans,tnames):
        if 'ki,kj' in tname or 'kj,ki' in tname:
            tspan_avg = tspan / max(1, nkptij if 'ji' in tname else nkptijswap)
            log.debug1('CPU time for ao2mo pass2 avg %12s  %9.2f sec, '
                       'wall time  %9.2f sec', tname, *tspan_avg)

    t0 = log.timer_debug1('ao2mo pass2', *t0)

    return out


def ao2mo_e2_Lij_kernel2(mydf, mo_coeffs, kpts, bvk_kmesh=None, out=None):
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
    log = logger.new_logger(mydf)
    verbose1 = mydf.verbose - 2

    nkpts = len(kpts)

    mo_coeff1, mo_coeff2 = mo_coeffs
    assert(len(mo_coeff1) == nkpts and len(mo_coeff2) == nkpts)

    nao = mo_coeff1[0].shape[0]
    naoaux = mydf.auxcell.nao_nr()
    if gamma_point(kpts):
        dtype = np.double
        dsize = 8
    else:
        dtype = np.complex128
        dsize = 16
    dtype = np.result_type(dtype, *mo_coeff1, *mo_coeff2)

    if out is None:
        out = np.empty((nkpts,nkpts), dtype=object)
    elif isinstance(out, np.ndarray):
        assert(out.shape == (nkpts,nkpts))
        assert(out.dtype == object)
    elif not isinstance(out, h5py.Group):
        raise TypeError('Input out must be np.ndarray or h5py.Group.')

    incore = isinstance(out, np.ndarray)
    if incore:
        hasdata = _hasdata_incore
        loaddata = _loaddata_incore
        writedata = _writedata_incore
        accumdata = _accumdata_incore
        log.debug1('transformed integrals will be held incore.')
    else:
        hasdata = _hasdata_outcore
        loaddata = _loaddata_outcore
        writedata = _writedata_outcore
        accumdata = _accumdata_outcore
        log.debug1('transformed integrals will be saved to specified h5py file.')

    kptij_lst = get_kptij_lst(kpts)
    nkptij = len(kptij_lst)
    uniq_q_loop = [x for x in loop_uniq_q(mydf, kptij_lst=kptij_lst, verbose=0)]
    uniq_kpts = [x[0] for x in uniq_q_loop]
    nkpts_uniq = len(uniq_kpts)
    nkptjmax = np.max([len(x[1]) for x in uniq_q_loop])
    nkptijswap = sum([1 for x in uniq_q_loop for kptj in x[1]
                      if _safe_member(kptj, kpts)!=_safe_member(kptj-x[0], kpts)])

# evaluate and invert j2c
    t0 = (logger.process_clock(), logger.perf_counter())
    kj2c = get_j2c(mydf, kpts=uniq_kpts, verbose=verbose1)
    kj2c_negative = [None] * nkpts_uniq
    kj2ctag = [None] * nkpts_uniq
    for k,kpt in enumerate(uniq_kpts):
        kj2c[k], kj2c_negative[k], kj2ctag[k] = cholesky_decomposed_metric(mydf, kj2c[k])
    t0 = log.timer_debug1('ao2mo j2c', *t0)

# buffer size
    nmo1s = np.array([mo.shape[1] for mo in mo_coeff1])
    nmo2s = np.array([mo.shape[1] for mo in mo_coeff2])
    size_Lij0 = int(lib.einsum('i,j->ij',nmo1s,nmo2s).sum())
    size_Lij = 1    # "intermediate" Lij
    if incore: size_Lij += nkpts**2
    size_Lij *= size_Lij0
    mem_Lij = size_Lij * dsize / 1e6

    mem_avail = mydf.max_memory - lib.current_memory()[0] - mem_Lij
    size_Lpqblk = (nkptij+nkptjmax+2)*naoaux    # add 1 for potential mem use in einsum
    mem_Lpqblk = size_Lpqblk * dsize / 1e6
    aopblksize = min(nao*nao, int(np.floor(mem_avail*0.7/mem_Lpqblk)))
    shranges = _guess_shell_ranges(mydf.cell, aopblksize, 's1')
    aopblksize = np.max([x[2] for x in shranges])
    pblksize = aopblksize // nao
    log.debug1('ao2mo mem_avail= %.2f MB  mem_Lij= %.2f MB  mem_Lpqblk= %.2f MB',
               mem_avail, mem_Lij, mem_Lpqblk)
    log.debug1('ao2mo aopblksize= %d  pblksize= %d  nblk= %d', aopblksize, pblksize,
               len(shranges))
    log.debug1('ao2mo shranges= %s', shranges)

    tspans = np.zeros((7,2))
    tnames = ['ki,kj fit  ', 'ki,kj xform', 'ki,kj write', 'kj,ki xform', 'kj,ki write', 'xform', 'j3c']
    t1_tock = logger.process_clock(), logger.perf_counter()

    p1 = 0
    for kcLpq in loop_j3c(mydf, kptij_lst=kptij_lst, aosym='s1', partition_iorj='i',
                          j3c_order='Lij', shranges=shranges, bvk_kmesh=bvk_kmesh,
                          verbose=verbose1):
        dp = kcLpq.shape[-1] // nao
        assert(dp*nao == kcLpq.shape[-1])
        p0 = p1
        p1 += dp

        kq = 0
        for kpt,adapted_kptjs,adapted_ji_idx in uniq_q_loop:
            j2c = kj2c[kq]
            j2ctag = kj2ctag[kq]
            for kptj,ji in zip(adapted_kptjs,adapted_ji_idx):
                kj = _safe_member(kptj, kpts)
                ki = _safe_member(kptj-kpt, kpts)

                tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                Lpq = scipy.linalg.solve_triangular(j2c, kcLpq[ji][0],
                                                    lower=True).reshape(naoaux,dp,nao)
                tock = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[0] += tock - tick
                mo1 = mo_coeff1[ki][p0:p1]
                mo2 = mo_coeff2[kj]
                nmo1 = mo1.shape[1]
                nmo2 = mo2.shape[1]
                Lij = lib.einsum('Lpq,pi,qj->Lij', Lpq, mo1.conj(), mo2)
                Lij = Lij.reshape(-1,nmo1,nmo2)
                tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[1] += tick - tock
                if hasdata(out,ki,kj):
                    accumdata(out,ki,kj,Lij)
                else:
                    writedata(out,ki,kj,Lij)
                Lij = None
                tock = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[2] += tock - tick

                if ki != kj:
                    mo1 = mo_coeff1[kj]
                    mo2 = mo_coeff2[ki][p0:p1]
                    nmo1 = mo1.shape[1]
                    nmo2 = mo2.shape[1]
                    Lji = lib.einsum('Lpq,pi,qj->Lij', Lpq, mo2.conj(), mo1).conj()
                    Lij = Lji.reshape(-1,nmo2,nmo1).transpose(0,2,1)
                    tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[3] += tick - tock
                    if hasdata(out,kj,ki):
                        accumdata(out,kj,ki,Lij)
                    else:
                        writedata(out,kj,ki,Lij)
                    tock = np.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[4] += tock - tick
                Lpq = Lij = Lji = None
            kq += 1

        t1_tick = t1_tock
        t1_tock = log.timer_debug1('ao2mo pass1 [%d:%d]'%(p0,p1), *t1_tick)
        tspans[6] += np.asarray(t1_tock) - np.asarray(t1_tick)

    tspans[5] = tspans[:5].sum(axis=0)
    tspans[6] -= tspans[5]
    for tspan,tname in zip(tspans,tnames):
        log.debug1('CPU time for ao2mo pass1     %12s  %9.2f sec, '
                   'wall time  %9.2f sec', tname, *tspan)
    for tspan,tname in zip(tspans,tnames):
        if 'ki,kj' in tname or 'kj,ki' in tname:
            tspan_avg = tspan / max(1, nkptij if 'ji' in tname else nkptijswap)
            log.debug1('CPU time for ao2mo pass1 avg %12s  %9.2f sec, '
                       'wall time  %9.2f sec', tname, *tspan_avg)

    return out

def ao2mo_e2_ijL_kernel1(mydf, mo_coeffs, kpts, bvk_kmesh=None, out=None):
    r''' (mu,nu|L) --> (i,j|L)

    Args:
        mo_coeffs (tuple or list):
            mo_coeffs = (C1_ks, C2_ks) where C1_ks and C2_ks are the mo coeff matrices
            for i and j for all kpts.
        out (numpy array of type "object" or h5py group):
            Where are the results stored?
            If numpy array, must be of type "object" and shape (nkpts,nkpts).
            If None, such a numpy array is created.
    '''
    log = logger.new_logger(mydf)
    verbose1 = mydf.verbose - 2

    nkpts = len(kpts)

    mo_coeff1, mo_coeff2 = mo_coeffs
    assert(len(mo_coeff1) == nkpts and len(mo_coeff2) == nkpts)

    nao = mo_coeff1[0].shape[0]
    naoaux = mydf.auxcell.nao_nr()
    if gamma_point(kpts):
        dtype = np.double
        dsize = 8
    else:
        dtype = np.complex128
        dsize = 16
    dtype = np.result_type(dtype, *mo_coeff1, *mo_coeff2)

    if out is None:
        out = np.empty((nkpts,nkpts), dtype=object)
    elif isinstance(out, np.ndarray):
        assert(out.shape == (nkpts,nkpts))
        assert(out.dtype == object)
    elif not isinstance(out, h5py.Group):
        raise TypeError('Input out must be np.ndarray or h5py.Group.')

    incore = isinstance(out, np.ndarray)
    if incore:
        hasdata = _hasdata_incore
        loaddata = _loaddata_incore
        writedata = _writedata_incore
        accumdata = _accumdata_incore
        log.debug1('transformed integrals will be held incore.')
    else:
        hasdata = _hasdata_outcore
        loaddata = _loaddata_outcore
        writedata = _writedata_outcore
        accumdata = _accumdata_outcore
        log.debug1('transformed integrals will be saved to specified h5py file.')

    kptij_lst = get_kptij_lst(kpts)
    nkptij = len(kptij_lst)
    uniq_q_loop = [x for x in loop_uniq_q(mydf, kptij_lst=kptij_lst, verbose=0)]
    uniq_kpts = [x[0] for x in uniq_q_loop]
    nkpts_uniq = len(uniq_kpts)
    nkptjmax = np.max([len(x[1]) for x in uniq_q_loop])
    nkptijswap = sum([1 for x in uniq_q_loop for kptj in x[1]
                      if _safe_member(kptj, kpts)!=_safe_member(kptj-x[0], kpts)])

# evaluate and invert j2c
    t0 = (logger.process_clock(), logger.perf_counter())
    kj2c = get_j2c(mydf, kpts=uniq_kpts, verbose=verbose1)
    kj2c_negative = [None] * nkpts_uniq
    kj2ctag = [None] * nkpts_uniq
    for k,kpt in enumerate(uniq_kpts):
        kj2c[k], kj2c_negative[k], kj2ctag[k] = cholesky_decomposed_metric(mydf, kj2c[k])
    t0 = log.timer_debug1('ao2mo j2c', *t0)

# buffer size
    nmo1s = np.array([mo.shape[1] for mo in mo_coeff1])
    nmo2s = np.array([mo.shape[1] for mo in mo_coeff2])
    size_Lij0 = int(lib.einsum('i,j->ij',nmo1s,nmo2s).sum())
    size_Lij = 1    # "intermediate" Lij
    if incore: size_Lij += nkpts**2
    size_Lij *= size_Lij0
    mem_Lij = size_Lij * dsize / 1e6

    mem_avail = mydf.max_memory - lib.current_memory()[0] - mem_Lij
    size_Lpqblk = (nkptij+nkptjmax+1)*naoaux    # add 1 for potential mem use in einsum
    mem_Lpqblk = size_Lpqblk * dsize / 1e6
    aopblksize = min(nao*nao, int(np.floor(mem_avail*0.7/mem_Lpqblk)))
    shranges = _guess_shell_ranges(mydf.cell, aopblksize, 's1')
    aopblksize = np.max([x[2] for x in shranges])
    pblksize = aopblksize // nao
    log.debug1('ao2mo mem_avail= %.2f MB  mem_Lij= %.2f MB  mem_Lpqblk= %.2f MB',
               mem_avail, mem_Lij, mem_Lpqblk)
    log.debug1('ao2mo aopblksize= %d  pblksize= %d  nblk= %d', aopblksize, pblksize,
               len(shranges))
    log.debug1('ao2mo shranges= %s', shranges)

    tspans = np.zeros((6,2))
    tnames = ['ki,kj xform', 'ki,kj write', 'kj,ki xform', 'kj,ki write', 'xform', 'j3c']

    t1_tock = logger.process_clock(), logger.perf_counter()

    p1 = 0
    for kcpqL in loop_j3c(mydf, kptij_lst=kptij_lst, aosym='s1', partition_iorj='i',
                          j3c_order='ijL', shranges=shranges, bvk_kmesh=bvk_kmesh,
                          verbose=verbose1):
        dp = kcpqL.shape[-2] // nao
        assert(dp*nao == kcpqL.shape[-2])
        p0 = p1
        p1 += dp

        for kpt,adapted_kptjs,adapted_ji_idx in uniq_q_loop:
            for kptj,ji in zip(adapted_kptjs,adapted_ji_idx):
                kj = _safe_member(kptj, kpts)
                ki = _safe_member(kptj-kpt, kpts)

                tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                pqL = kcpqL[ji][0].reshape(dp,nao,naoaux)
                mo1 = mo_coeff1[ki][p0:p1]
                mo2 = mo_coeff2[kj]
                nmo1 = mo1.shape[1]
                nmo2 = mo2.shape[1]
                ijL = lib.einsum('pqL,pi,qj->ijL', pqL, mo1.conj(), mo2)
                tock = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[0] += tock - tick
                ijL = ijL.reshape(nmo1,nmo2,-1)
                if hasdata(out,ki,kj):
                    accumdata(out,ki,kj,ijL)
                else:
                    writedata(out,ki,kj,ijL)
                ijL = None
                tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[1] += tick - tock

                if ki != kj:
                    mo1 = mo_coeff1[kj]
                    mo2 = mo_coeff2[ki][p0:p1]
                    nmo1 = mo1.shape[1]
                    nmo2 = mo2.shape[1]
                    jiL = lib.einsum('pqL,pi,qj->ijL', pqL, mo2.conj(), mo1).conj()
                    tock = np.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[2] += tock - tick
                    ijL = jiL.reshape(nmo2,nmo1,-1).transpose(1,0,2)
                    if hasdata(out,kj,ki):
                        accumdata(out,kj,ki,ijL)
                    else:
                        writedata(out,kj,ki,ijL)
                    tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[3] += tick - tock
                pqL = ijL = jiL = None
        t1_tick = t1_tock
        t1_tock = log.timer_debug1('ao2mo pass1 [%d:%d]'%(p0,p1), *t1_tick)
        tspans[5] += np.asarray(t1_tock) - np.asarray(t1_tick)

    tspans[4] = tspans[:4].sum(axis=0)
    tspans[5] -= tspans[4]
    for tspan,tname in zip(tspans,tnames):
        log.debug1('CPU time for ao2mo pass1     %12s  %9.2f sec, '
                   'wall time  %9.2f sec', tname, *tspan)
    for tspan,tname in zip(tspans,tnames):
        if 'ki,kj' in tname or 'kj,ki' in tname:
            tspan_avg = tspan / max(1, nkptij if 'ji' in tname else nkptijswap)
            log.debug1('CPU time for ao2mo pass1 avg %12s  %9.2f sec, '
                       'wall time  %9.2f sec', tname, *tspan_avg)

    t0 = log.timer_debug1('ao2mo pass1', *t0)

    tspans = np.zeros((6,2))
    tnames = ['ki,kj  load','ki,kj solve','ki,kj write',
              'kj,ki  load','kj,ki solve','kj,ki write']

    kq = 0
    for kpt,adapted_kptjs,adapted_ji_idx in uniq_q_loop:
        j2c = kj2c[kq]
        j2ctag = kj2ctag[kq]
        for kptj,ji in zip(adapted_kptjs,adapted_ji_idx):
            tick = np.asarray((logger.process_clock(), logger.perf_counter()))
            kj = _safe_member(kptj, kpts)
            ki = _safe_member(kptj-kpt, kpts)
            ijL = loaddata(out,ki,kj)
            tock = np.asarray((logger.process_clock(), logger.perf_counter()))
            tspans[0] += tock - tick
            ijL = scipy.linalg.solve_triangular(j2c, ijL.T, lower=True).T
            tick = np.asarray((logger.process_clock(), logger.perf_counter()))
            tspans[1] += tick - tock
            writedata(out,ki,kj,ijL)
            tock = np.asarray((logger.process_clock(), logger.perf_counter()))
            tspans[2] += tock - tick
            ijL = None
            if ki != kj:
                ijL = loaddata(out,kj,ki)
                tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[3] += tick - tock
                ijL = scipy.linalg.solve_triangular(j2c.conj(), ijL.T, lower=True).T
                tock = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[4] += tock - tick
                writedata(out,kj,ki,ijL)
                tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[5] += tick - tock
                ijL = None
        kq += 1

    for tspan,tname in zip(tspans,tnames):
        log.debug1('CPU time for ao2mo pass2     %12s  %9.2f sec, '
                   'wall time  %9.2f sec', tname, *tspan)
    for tspan,tname in zip(tspans,tnames):
        if 'ki,kj' in tname or 'kj,ki' in tname:
            tspan_avg = tspan / max(1, nkptij if 'ji' in tname else nkptijswap)
            log.debug1('CPU time for ao2mo pass2 avg %12s  %9.2f sec, '
                       'wall time  %9.2f sec', tname, *tspan_avg)

    t0 = log.timer_debug1('ao2mo pass2', *t0)

    return out

def ao2mo_e2_ijL_kernel2(mydf, mo_coeffs, kpts, bvk_kmesh=None, out=None):
    r''' (mu,nu|L) --> (i,j|L)

    Args:
        mo_coeffs (tuple or list):
            mo_coeffs = (C1_ks, C2_ks) where C1_ks and C2_ks are the mo coeff matrices
            for i and j for all kpts.
        out (numpy array of type "object" or h5py group):
            Where are the results stored?
            If numpy array, must be of type "object" and shape (nkpts,nkpts).
            If None, such a numpy array is created.
    '''
    log = logger.new_logger(mydf)
    verbose1 = mydf.verbose - 2

    nkpts = len(kpts)

    mo_coeff1, mo_coeff2 = mo_coeffs
    assert(len(mo_coeff1) == nkpts and len(mo_coeff2) == nkpts)

    nao = mo_coeff1[0].shape[0]
    naoaux = mydf.auxcell.nao_nr()
    if gamma_point(kpts):
        dtype = np.double
        dsize = 8
    else:
        dtype = np.complex128
        dsize = 16
    dtype = np.result_type(dtype, *mo_coeff1, *mo_coeff2)

    if out is None:
        out = np.empty((nkpts,nkpts), dtype=object)
    elif isinstance(out, np.ndarray):
        assert(out.shape == (nkpts,nkpts))
        assert(out.dtype == object)
    elif not isinstance(out, h5py.Group):
        raise TypeError('Input out must be np.ndarray or h5py.Group.')

    incore = isinstance(out, np.ndarray)
    if incore:
        hasdata = _hasdata_incore
        loaddata = _loaddata_incore
        writedata = _writedata_incore
        accumdata = _accumdata_incore
        log.debug1('transformed integrals will be held incore.')
    else:
        hasdata = _hasdata_outcore
        loaddata = _loaddata_outcore
        writedata = _writedata_outcore
        accumdata = _accumdata_outcore
        log.debug1('transformed integrals will be saved to specified h5py file.')

    kptij_lst = get_kptij_lst(kpts)
    nkptij = len(kptij_lst)
    uniq_q_loop = [x for x in loop_uniq_q(mydf, kptij_lst=kptij_lst, verbose=0)]
    uniq_kpts = [x[0] for x in uniq_q_loop]
    nkpts_uniq = len(uniq_kpts)
    nkptjmax = np.max([len(x[1]) for x in uniq_q_loop])
    nkptijswap = sum([1 for x in uniq_q_loop for kptj in x[1]
                      if _safe_member(kptj, kpts)!=_safe_member(kptj-x[0], kpts)])

# evaluate and invert j2c
    t0 = (logger.process_clock(), logger.perf_counter())
    kj2c = get_j2c(mydf, kpts=uniq_kpts, verbose=verbose1)
    kj2c_negative = [None] * nkpts_uniq
    kj2ctag = [None] * nkpts_uniq
    for k,kpt in enumerate(uniq_kpts):
        kj2c[k], kj2c_negative[k], kj2ctag[k] = cholesky_decomposed_metric(mydf, kj2c[k])
    t0 = log.timer_debug1('ao2mo j2c', *t0)

# buffer size
    nmo1s = np.array([mo.shape[1] for mo in mo_coeff1])
    nmo2s = np.array([mo.shape[1] for mo in mo_coeff2])
    size_Lij0 = int(lib.einsum('i,j->ij',nmo1s,nmo2s).sum())
    size_Lij = 2    # "intermediate" Lij
    if incore: size_Lij += nkpts**2
    size_Lij *= size_Lij0
    mem_Lij = size_Lij * dsize / 1e6

    mem_avail = mydf.max_memory - lib.current_memory()[0] - mem_Lij
    size_Lpqblk = (nkptij+nkptjmax+2)*naoaux    # add 1 for potential mem use in einsum
    mem_Lpqblk = size_Lpqblk * dsize / 1e6
    aopblksize = min(nao*nao, int(np.floor(mem_avail*0.7/mem_Lpqblk)))
    shranges = _guess_shell_ranges(mydf.cell, aopblksize, 's1')
    aopblksize = np.max([x[2] for x in shranges])
    pblksize = aopblksize // nao
    log.debug1('ao2mo mem_avail= %.2f MB  mem_Lij= %.2f MB  mem_Lpqblk= %.2f MB',
               mem_avail, mem_Lij, mem_Lpqblk)
    log.debug1('ao2mo aopblksize= %d  pblksize= %d  nblk= %d', aopblksize, pblksize,
               len(shranges))
    log.debug1('ao2mo shranges= %s', shranges)

    tspans = np.zeros((7,2))
    tnames = ['ki,kj fit  ', 'ki,kj xform', 'ki,kj write', 'kj,ki xform', 'kj,ki write', 'xform', 'j3c']
    t1_tock = logger.process_clock(), logger.perf_counter()

    p1 = 0
    for kcpqL in loop_j3c(mydf, kptij_lst=kptij_lst, aosym='s1', partition_iorj='i',
                          j3c_order='ijL', shranges=shranges, bvk_kmesh=bvk_kmesh,
                          verbose=verbose1):
        dp = kcpqL.shape[-2] // nao
        assert(dp*nao == kcpqL.shape[-2])
        p0 = p1
        p1 += dp

        kq = 0
        for kpt,adapted_kptjs,adapted_ji_idx in uniq_q_loop:
            j2c = kj2c[kq]
            j2ctag = kj2ctag[kq]
            for kptj,ji in zip(adapted_kptjs,adapted_ji_idx):
                kj = _safe_member(kptj, kpts)
                ki = _safe_member(kptj-kpt, kpts)

                tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                pqL = scipy.linalg.solve_triangular(j2c, kcpqL[ji][0].T,
                                                    lower=True).T.reshape(dp,nao,naoaux)
                tock = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[0] += tock - tick
                mo1 = mo_coeff1[ki][p0:p1]
                mo2 = mo_coeff2[kj]
                nmo1 = mo1.shape[1]
                nmo2 = mo2.shape[1]
                ijL = lib.einsum('pqL,pi,qj->ijL', pqL, mo1.conj(), mo2)
                ijL = ijL.reshape(nmo1,nmo2,-1)
                tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[1] += tick - tock
                if hasdata(out,ki,kj):
                    accumdata(out,ki,kj,ijL)
                else:
                    writedata(out,ki,kj,ijL)
                ijL = None
                tock = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[2] += tock - tick

                if ki != kj:
                    mo1 = mo_coeff1[kj]
                    mo2 = mo_coeff2[ki][p0:p1]
                    nmo1 = mo1.shape[1]
                    nmo2 = mo2.shape[1]
                    jiL = lib.einsum('pqL,pi,qj->ijL', pqL, mo2.conj(), mo1).conj()
                    ijL = np.asarray(jiL.reshape(nmo2,nmo1,-1).transpose(1,0,2),
                                     order='C')
                    tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[3] += tick - tock
                    if hasdata(out,kj,ki):
                        accumdata(out,kj,ki,ijL)
                    else:
                        writedata(out,kj,ki,ijL)
                    tock = np.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[4] += tock - tick
                pqL = ijL = jiL = None
            kq += 1

        t1_tick = t1_tock
        t1_tock = log.timer_debug1('ao2mo pass1 [%d:%d]'%(p0,p1), *t1_tick)
        tspans[6] += np.asarray(t1_tock) - np.asarray(t1_tick)

    tspans[5] = tspans[:5].sum(axis=0)
    tspans[6] -= tspans[5]
    for tspan,tname in zip(tspans,tnames):
        log.debug1('CPU time for ao2mo pass1     %12s  %9.2f sec, '
                   'wall time  %9.2f sec', tname, *tspan)
    for tspan,tname in zip(tspans,tnames):
        if 'ki,kj' in tname or 'kj,ki' in tname:
            tspan_avg = tspan / max(1, nkptij if 'ji' in tname else nkptijswap)
            log.debug1('CPU time for ao2mo pass1 avg %12s  %9.2f sec, '
                       'wall time  %9.2f sec', tname, *tspan_avg)

    return out

def ao2mo_e2(mydf, mo_coeffs, kpts=None, j3c_order='Lij', kernel=AO2MO_KERNEL, out=None):
    log = logger.new_logger(mydf)

    assert(j3c_order in ['Lij','ijL'])
    assert(kernel in [1,2])
    if j3c_order == 'Lij':
        if kernel == 1:
            fao2mo = ao2mo_e2_Lij_kernel1
        else:
            fao2mo = ao2mo_e2_Lij_kernel2
    else:
        if kernel == 1:
            fao2mo = ao2mo_e2_ijL_kernel1
        else:
            fao2mo = ao2mo_e2_ijL_kernel2
    log.debug1('Using fao2mo kernel ao2mo_e2_%s_kernel%d', j3c_order, kernel)

    if isinstance(mydf.use_bvk, bool):
        use_bvk_R = use_bvk_G = mydf.use_bvk
    else:
        use_bvk_R,  use_bvk_G = mydf.use_bvk
    if use_bvk_R or use_bvk_G:
        from pyscf.pbc.df.rsdf_direct_helper import kpts_to_kmesh
        bvk_kmesh0 = kpts_to_kmesh(mydf.cell, kpts)
        bvk_kmesh = [bvk_kmesh0 if use_bvk_R else None,
                     bvk_kmesh0 if use_bvk_G else None]
    else:
        bvk_kmesh = None
    log.debug1('Using bvk_kmesh= %s', bvk_kmesh)

    t0 = (logger.process_clock(), logger.perf_counter())
    out = fao2mo(mydf, mo_coeffs, kpts=kpts, bvk_kmesh=bvk_kmesh, out=out)
    log.timer('RSDF direct ao2mo', *t0)

    return out



def _hasdata_incore(out,k1,k2):
    return out[k1,k2] is not None
def _loaddata_incore(out,k1,k2):
    return out[k1,k2]
def _writedata_incore(out,k1,k2,L):
    out[k1,k2] = L
def _accumdata_incore(out,k1,k2,L):
    out[k1,k2] += L
def _hasdata_outcore(out,k1,k2):
    kstr = '%d,%d'%(k1,k2)
    return kstr in out
def _loaddata_outcore(out,k1,k2):
    kstr = '%d,%d'%(k1,k2)
    return out[kstr][()]
def _writedata_outcore(out,k1,k2,L):
    kstr = '%d,%d'%(k1,k2)
    if kstr not in out:
        out[kstr] = L
    elif out[kstr].shape == L.shape:
        out[kstr][()] = L
    else:
        del out[kstr]
        out[kstr] = L
def _accumdata_outcore(out,k1,k2,L):
    kstr = '%d,%d'%(k1,k2)
    out[kstr][()] += L


if __name__ == '__main__':
    atom = 'He 0 0 0; He 1 0 0'
    a = np.eye(3) * 3
    basis = 'cc-pvdz'

    from pyscf.pbc import gto, scf, df
    cell = gto.Cell(atom=atom, a=a, basis=basis)
    cell.build()

    kmesh = [2,3,1]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.kernel()

    orbocc = [mo[:,occ>1e-10] for mo,occ in zip(mf.mo_coeff,mf.mo_occ)]
    orbvir = [mo[:,occ<=1e-10] for mo,occ in zip(mf.mo_coeff,mf.mo_occ)]
    mo_coeffs = (orbocc, orbvir)
    from pyscf.pbc.df.rsdf_ao2mo import ao2mo_e2 as ao2mo_e2_ref
    mf.with_df.verbose = 6
    kkLovref = ao2mo_e2_ref(mf.with_df, mo_coeffs)

    # mydf = df.RSDF(cell, kpts)
    # mydf.direct = True
    # mydf.verbose = 6
    # mydf.build()
    # for kernel in [1,2]:
    #     f = lib.H5TmpFile()
    #     kkLov = f.create_group('kkLov')
    #     # kkLov = None
    #     kkLov = ao2mo_e2(mydf, mo_coeffs, kpts, j3c_order='Lij', kernel=kernel, out=kkLov)
    #     for k1 in range(nkpts):
    #         for k2 in range(nkpts):
    #             if isinstance(kkLov, np.ndarray):
    #                 print(kkLov[k1,k2].data.f_contiguous, kkLovref[k1,k2].data.c_contiguous)
    #                 err_real = abs(kkLov[k1,k2].real - kkLovref[k1,k2].real).max()
    #                 err_imag = abs(kkLov[k1,k2].imag - kkLovref[k1,k2].imag).max()
    #             else:
    #                 kstr = '%d,%d'%(k1,k2)
    #                 print(kkLov[kstr][()].data.c_contiguous,
    #                       kkLovref[k1,k2].data.c_contiguous)
    #                 err_real = abs(kkLov[kstr][()].real - kkLovref[k1,k2].real).max()
    #                 err_imag = abs(kkLov[kstr][()].imag - kkLovref[k1,k2].imag).max()
    #             print(f'{k1:2d} {k2:2d}  {err_real:.3e}  {err_imag:.3e}')
    #     if isinstance(kkLov, h5py.Group):
    #         f.close()

    mydf = df.RSDF(cell, kpts)
    mydf.direct = True
    mydf.verbose = 6
    mydf.build()
    for kernel in [1,2]:
        f = lib.H5TmpFile()
        kkLov = f.create_group('kkLov')
        # kkLov = None
        kkLov = ao2mo_e2(mydf, mo_coeffs, kpts, j3c_order='ijL', kernel=kernel, out=kkLov)
        for k1 in range(nkpts):
            for k2 in range(nkpts):
                if isinstance(kkLov, np.ndarray):
                    print(kkLov[k1,k2].data.c_contiguous, kkLovref[k1,k2].data.c_contiguous)
                    err_real = abs(kkLov[k1,k2].real.transpose(2,0,1) -
                                   kkLovref[k1,k2].real).max()
                    err_imag = abs(kkLov[k1,k2].imag.transpose(2,0,1) -
                                   kkLovref[k1,k2].imag).max()
                else:
                    kstr = '%d,%d'%(k1,k2)
                    print(kkLov[kstr][()].data.c_contiguous,
                          kkLovref[k1,k2].data.c_contiguous)
                    err_real = abs(kkLov[kstr][()].real.transpose(2,0,1) -
                                   kkLovref[k1,k2].real).max()
                    err_imag = abs(kkLov[kstr][()].imag.transpose(2,0,1) -
                                   kkLovref[k1,k2].imag).max()
                print(f'{k1:2d} {k2:2d}  {err_real:.3e}  {err_imag:.3e}')
        if isinstance(kkLov, h5py.Group):
            f.close()
