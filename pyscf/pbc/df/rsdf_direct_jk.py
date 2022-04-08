r''' Integral-direct implementation of get_jk
'''

import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member, unique
from pyscf import __config__

EIGH_DM_THRESH = getattr(__config__, 'pbc_gto_df_rsdf_jk_direct_eigh_dm_thresh', 1e-10)
REAL = np.float64
COMPLEX = np.complex128

r''' Needed functions
    - get_j  : single kpt
    - get_k  : single kpt
    - get_j_kpts
    - get_k_kpts
'''

''' To-do
[x] block get_j_kpts
[x] make get_j_kpts enjoy aosym='s2'
[x] branching real and complex cases in get_k_kpts (e.g., solve_triangular)
[x] handle real mo_coeff in get_k_kpts
[x] support nset > 1 for get_k_kpts
[x] support bvk_kmesh
[x] make prescreening precomputeable
[x] get_k gamma uses ijL
[x] get_k complex swap ij and ji
'''


from pyscf.pbc.df.df_jk import (_format_dms, _format_jks, _format_kpts_band,
                                _ewald_exxdiv_for_G0, zdotNN, zdotCN, zdotNC)
from pyscf.pbc.df.rsdf_direct_helper import (
                                get_j2c, get_j3c, cholesky_decomposed_metric,
                                get_kptij_lst, loop_uniq_q, loop_j3c)


def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
               bvk_kmesh=None):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    verbose1 = mydf.verbose - 2
    t1 = (logger.process_clock(), logger.perf_counter())

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    if mydf.auxcell is None:
        mydf.build()
    naux = mydf.auxcell.nao_nr()
    nao_pair = nao * (nao+1) // 2

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    j_real = gamma_point(kpts_band) and not np.iscomplexobj(dms)

    dmsR = np.asarray(dms.real.transpose(0,1,3,2).reshape(nset,nkpts,nao**2), order='C')
    dmsI = np.asarray(dms.imag.transpose(0,1,3,2).reshape(nset,nkpts,nao**2), order='C')

# allocate memory
    rhoR = np.zeros((nset,naux))
    rhoI = np.zeros((nset,naux))
    vjR = np.zeros((nset,nband,nao_pair))
    vjI = np.zeros((nset,nband,nao_pair))

# step 1: \sum_kj \sum_{mu,nu} (L|mu,nu)^{kj,kj} dm_{mu,nu}^kj -> rho_L
    kptii_lst = np.repeat(kpts,2,axis=0).reshape(nkpts,2,3)
    j3c_dtype, j3c_dsize = (REAL,8) if is_zero(kptii_lst) else (COMPLEX,16)
    j3c_real = j3c_dtype == REAL
    mem_avail = mydf.max_memory - lib.current_memory()[0]
    log.debug1('get_j pass1 mem_avail= %.1f MB', mem_avail)
    blksize = min(nao*nao, mem_avail*0.7e6 / (2*nkpts*naux*j3c_dsize))
    shranges = _guess_shell_ranges(mydf.cell, blksize, 's1')
    log.debug1('get_j pass1 blksize= %s  shranges= %s', blksize, shranges)
    blksize = np.max([x[2] for x in shranges])
    bufR = np.empty(naux*blksize, dtype=REAL)
    bufI = np.empty(naux*blksize, dtype=REAL)
    p1 = 0
    for kcpqL in loop_j3c(mydf, kptij_lst=kptii_lst, aosym='s1', partition_iorj='i',
                          j3c_order='ijL', shranges=shranges, bvk_kmesh=bvk_kmesh,
                          verbose=verbose1):
        dp = kcpqL.shape[-2]
        p0 = p1
        p1 += dp
        pqLR = np.ndarray((dp,naux), dtype=REAL, buffer=bufR)
        if not j3c_real:
            pqLI = np.ndarray((dp,naux), dtype=REAL, buffer=bufI)
        for k,kpt in enumerate(kpts):
            pqLR[:] = kcpqL[k][0].real
            rhoR += lib.dot(dmsR[:,k,p0:p1], pqLR)
            rhoI += lib.dot(dmsI[:,k,p0:p1], pqLR)
            if not j3c_real:
                pqLI[:] = kcpqL[k][0].imag
                rhoR -= lib.dot(dmsI[:,k,p0:p1], pqLI)
                rhoI += lib.dot(dmsR[:,k,p0:p1], pqLI)
        pqLR = pqLI = kcpqL = None
    bufR = bufI = None

    weight = 1./nkpts
    rhoR *= weight
    rhoI *= weight
    t1 = log.timer_debug1('get_j pass1   ', *t1)

# setp 2: j2v inv
    j2c = get_j2c(mydf, kpts=np.zeros((1,3)), verbose=verbose1)[0]
    j2c, j2c_negative, j2ctag = cholesky_decomposed_metric(mydf, j2c)
    if j2ctag == 'CD':
        # if b is F_CONTIGUOUS and j2c is real, b will be overwritten
        if rhoR.T.flags['F_CONTIGUOUS'] and j2c.dtype == REAL:
            scipy.linalg.solve_triangular(j2c, rhoR.T, lower=True, overwrite_b=True)
            scipy.linalg.solve_triangular(j2c, rhoR.T, lower=True, overwrite_b=True,
                                          trans=1)
        else:
            rhoR = scipy.linalg.solve_triangular(j2c, rhoR.T, lower=True)
            rhoR = scipy.linalg.solve_triangular(j2c, rhoR, lower=True, trans=1).T
        if rhoI.T.flags['F_CONTIGUOUS'] and j2c.dtype == REAL:
            scipy.linalg.solve_triangular(j2c, rhoI.T, lower=True, overwrite_b=True)
            scipy.linalg.solve_triangular(j2c, rhoI.T, lower=True, overwrite_b=True,
                                          trans=1)
        else:
            rhoI = scipy.linalg.solve_triangular(j2c, rhoI.T, lower=True)
            rhoI = scipy.linalg.solve_triangular(j2c, rhoI, lower=True, trans=1).T
    else:
        rhoR = lib.dot(lib.dot(rhoR, j2c.T.conj()), j2c)
        rhoI = lib.dot(lib.dot(rhoI, j2c.T.conj()), j2c)
    t1 = log.timer_debug1('get_j j2c_cntr', *t1)

# step 3: vj_{pq}^{ki} = \sum_{L} (L|pq)^{ki,ki} rho_L
    kptbandii_lst = np.repeat(kpts_band,2,axis=0).reshape(nband,2,3)
    mem_avail = mydf.max_memory - lib.current_memory()[0]
    log.debug1('get_j pass2 mem_avail= %.1f MB', mem_avail)
    blksize = min(nao*(nao+1)//2, mem_avail*0.7e6 / (2*nband*naux*j3c_dsize))
    shranges = _guess_shell_ranges(mydf.cell, blksize, 's2')
    log.debug1('get_j pass2 blksize= %s  shranges= %s', blksize, shranges)
    blksize = np.max([x[2] for x in shranges])
    bufR = np.empty(naux*blksize, dtype=REAL)
    bufI = np.empty(naux*blksize, dtype=REAL)
    p1 = 0
    for kcpqL in loop_j3c(mydf, kptij_lst=kptbandii_lst, aosym='s2', partition_iorj='i',
                          j3c_order='ijL', shranges=shranges, bvk_kmesh=bvk_kmesh,
                          verbose=verbose1):
        dp = kcpqL.shape[-2]
        p0 = p1
        p1 += dp
        pqLR = np.ndarray((naux,dp), dtype=REAL, buffer=bufR)
        pqLI = np.ndarray((naux,dp), dtype=REAL, buffer=bufI)
        for k,kpt in enumerate(kpts_band):
            pqLR = np.asarray(kcpqL[k][0].real, order='C')
            if not is_zero(kpt):
                pqLI = np.asarray(kcpqL[k][0].imag, order='C')
            vjR[:,k,p0:p1] += lib.dot(rhoR, pqLR.T)
            if not j_real:
                vjI[:,k,p0:p1] += lib.dot(rhoI, pqLR.T)
                if not is_zero(kpt):
                    vjR[:,k,p0:p1] -= lib.dot(rhoI, pqLI.T)
                    vjI[:,k,p0:p1] += lib.dot(rhoR, pqLI.T)
        pqLR = pqLI = kcpqL = None
    bufR = bufI = None
    t1 = log.timer_debug1('get_j pass2   ', *t1)

# post-proc
    if j_real:
        vj_kpts = vjR
    else:
        vj_kpts = vjR + vjI*1j
    vj_kpts = lib.unpack_tril(vj_kpts.reshape(-1,nao_pair))

    return _format_jks(vj_kpts, dm_kpts, input_band, kpts)


def _safe_member(q, qs):
    idxs = member(q, qs)
    if len(idxs) != 1:
        raise RuntimeError
    return idxs[0]
def get_k_kpts_gamma(mydf, smo):
    t1 = (logger.process_clock(), logger.perf_counter())

    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    verbose1 = mydf.verbose - 2

    nset = len(smo)
    nao = smo[0].shape[0]
    naux = mydf.auxcell.nao_nr()
    nmomax = np.max([mo.shape[1] for mo in smo])

    j3c_dtype,j3c_dsize = REAL,8
    vs_dtype,vs_dsize = REAL,8

    vs = np.zeros((nset,nao,nao))

    j2c = get_j2c(mydf, kpts=np.zeros((1,3)), verbose=verbose1)[0]
    j2c, j2c_negative, j2ctag = cholesky_decomposed_metric(mydf, j2c)
    t1 = log.timer_debug1('get_k_kpts j2c', *t1)

# estimate minimum memory requirement for j3c
    ao_loc = mydf.cell.ao_loc_nr()
    naoshlmax = np.max(ao_loc[1:] - ao_loc[:-1])
    j3cblksizemin = 2*naux*naoshlmax*nao
    mem_j3cblk = j3cblksizemin*j3c_dsize/1e6

# buffer for kXip and kYip
    mem_avail = mydf.max_memory - lib.current_memory()[0] - mem_j3cblk
    XYblksizemin = (nset+2)*naux*nao    # add 2 for Lpi, Xpi
    mem_XYblk = XYblksizemin*vs_dsize/1e6
    nmoblksize = min(nmomax, int(np.floor(mem_avail*0.7/mem_XYblk)))
    log.debug1('get_k mem_avail= %.2f MB  mem_XYblk= %.2f MB', mem_avail, mem_XYblk)
    log.debug1('get_k nmomax= %d  nmoblksize= %d  nblk= %d', nmomax, nmoblksize,
               nmomax//nmoblksize+nmomax%nmoblksize>0)
    if nmoblksize < 1:
        mem_need = mem_XYblk + mem_j3cblk
        log.error('Caching (L|[p]q) and (L|p[i]) needs at least %.1f MB of memory, '
                  'which exceeds the available memory %.1f MB', mem_need, mem_avail)
        raise MemoryError
    buf_sXip = np.empty((nset,naux,nao,nmoblksize), dtype=REAL)
    buf_Xip = np.empty(naux*nao*nmoblksize, dtype=REAL)

# shranges for j3c
    mem_avail -= mem_XYblk*nmoblksize
    j3cblksize = (2+1)*naux # add 2 for Lpq
    mem_j3cblk = j3cblksize*j3c_dsize/1e6
    aopblksize = min(nao*nao, mem_avail*0.7/mem_j3cblk)
    shranges = _guess_shell_ranges(mydf.cell, aopblksize, 's1')
    aopblksize = np.max([x[2] for x in shranges])
    pblksize = aopblksize // nao
    log.debug1('get_k mem_avail= %.2f MB  memj3cblk= %.2f MB', mem_avail, mem_j3cblk)
    log.debug1('get_k aopblksize= %d  pblksize= %d  nblk= %d', aopblksize, pblksize,
               len(shranges))
    log.debug1('get_k shranges= %s', shranges)
    buf_Lpi = np.empty(naux*pblksize*nmoblksize, dtype=REAL)

    for i0,i1 in lib.prange(0,nmomax,nmoblksize):
        di = i1 - i0

        spiX = np.ndarray((nset,nao,di,naux),dtype=REAL,buffer=buf_sXip)

        q1 = 0
        for kcpqL in loop_j3c(mydf, kptij_lst=np.zeros((1,2,3)), aosym='s1',
                              j3c_order='ijL', partition_iorj='j', shranges=shranges,
                              verbose=verbose1):
            dq = kcpqL.shape[-2] // nao
            assert(dq*nao == kcpqL.shape[-2])
            q0 = q1
            q1 += dq

            pqL = kcpqL[0][0].reshape(nao,-1)
            iqL = np.ndarray((di,naux*dq), dtype=REAL, buffer=buf_Lpi)
            for iset in range(nset):
                mo = smo[iset][:,i0:i1]
                lib.ddot(mo.T, pqL, c=iqL)
                spiX[iset,q0:q1] = iqL.reshape(di,dq,naux).transpose(1,0,2)
            pqL = iqL = kcpqL = None

        t1 = log.timer_debug1('get_k_kpts occblk [%d:%d] pass 1'%(i0,i1), *t1)

        if j2ctag == 'CD':
            piX = np.ndarray((nao*di,naux), dtype=REAL, buffer=buf_Xip)
            for iset in range(nset):
                piX[:] = spiX[iset].reshape(-1,naux)
                scipy.linalg.solve_triangular(j2c, piX.T, lower=True, overwrite_b=True)
                lib.ddot(piX.reshape(nao,-1), piX.reshape(nao,-1).T, c=vs[iset], beta=1)
        else:
            piX = np.ndarray((nao*di,naux), dtype=REAL, buffer=buf_Xip)
            for iset in range(nset):
                lib.ddot(spiX[iset].reshape(-1,naux), j2c.T, c=piX)
                lib.ddot(piX.reshape(nao,-1), piX.reshape(nao,-1).T, c=vs[iset], beta=1)

        piX = spiX = None

        t1 = log.timer_debug1('get_k_kpts occblk [%d:%d] pass 2'%(i0,i1), *t1)

    vk_kpts = vs.reshape((nset,1,nao,nao))

    return vk_kpts
def get_k_kpts_complex(mydf, skmoR, skmoI, kpts, bvk_kmesh=None):
    t1 = (logger.process_clock(), logger.perf_counter())

    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    verbose1 = mydf.verbose - 2

    nset = len(skmoR)
    nkpts = len(kpts)
    nband = nkpts
    nao = skmoR[0][0].shape[0]
    naux = mydf.auxcell.nao_nr()
    nmomax = np.max([moR.shape[1] for kmoR in skmoR for moR in kmoR])

    j3c_dtype,j3c_dsize = COMPLEX,16
    vk_dtype,vk_dsize = COMPLEX,16

    vkR = np.zeros((nset,nband,nao,nao))
    vkI = np.zeros((nset,nband,nao,nao))

    kptij_lst = get_kptij_lst(kpts)
    nkptij = len(kptij_lst)
    uniq_q_loop = [x for x in loop_uniq_q(mydf, kptij_lst=kptij_lst, verbose=0)]
    uniq_kpts = [x[0] for x in uniq_q_loop]
    nkpts_uniq = len(uniq_kpts)
    nkptjmax = np.max([len(x[1]) for x in uniq_q_loop])
    nkptijswap = sum([1 for x in uniq_q_loop for kptj in x[1]
                      if _safe_member(kptj, kpts)!=_safe_member(kptj-x[0], kpts)])

# evaluate and invert j2c
    kj2c = get_j2c(mydf, kpts=uniq_kpts, verbose=verbose1)
    kj2c_negative = [None] * nkpts_uniq
    kj2ctag = [None] * nkpts_uniq
    for k,kpt in enumerate(uniq_kpts):
        kj2c[k], kj2c_negative[k], kj2ctag[k] = cholesky_decomposed_metric(mydf, kj2c[k])

    t1 = log.timer_debug1('get_k_kpts j2c', *t1)

# estimate minimum memory requirement for j3c
    ao_loc = mydf.cell.ao_loc_nr()
    naoshlmax = np.max(ao_loc[1:] - ao_loc[:-1])
    j3cblksizemin = (nkptij+nkptjmax)*naux*naoshlmax*nao
    mem_j3cblk = j3cblksizemin*j3c_dsize/1e6

# buffer for kXip and kYip
    mem_avail = mydf.max_memory - lib.current_memory()[0] - mem_j3cblk
    XYblksizemin = (nset*(nkptij+nkptijswap)+2)*naux*nao    # add 2 for Lpi, Xpi
    mem_XYblk = XYblksizemin*vk_dsize/1e6
    nmoblksize = min(nmomax, int(np.floor(mem_avail*0.7/mem_XYblk)))
    log.debug1('get_k mem_avail= %.2f MB  mem_XYblk= %.2f MB', mem_avail, mem_XYblk)
    log.debug1('get_k nmomax= %d  nmoblksize= %d  nblk= %d', nmomax, nmoblksize,
               nmomax//nmoblksize+(1 if nmomax%nmoblksize>0 else 0))
    if nmoblksize < 1:
        mem_need = mem_XYblk + mem_j3cblk
        log.error('Caching (L|[p]q) and (L|p[i]) needs at least %.1f MB of memory, '
                  'which exceeds the available memory %.1f MB', mem_need, mem_avail)
        raise MemoryError
    buf_kXipR = np.empty((nset,nkptij,naux,nao,nmoblksize), dtype=REAL)
    buf_kYipR = np.empty((nset,nkptijswap,naux,nao,nmoblksize), dtype=REAL)
    buf_LqiR = np.empty(naux*nao*nmoblksize, dtype=REAL)
    buf_kXipI = np.empty((nset,nkptij,naux,nao,nmoblksize), dtype=REAL)
    buf_kYipI = np.empty((nset,nkptijswap,naux,nao,nmoblksize), dtype=REAL)
    buf_LqiI = np.empty(naux*nao*nmoblksize, dtype=REAL)
    buf_Xiq = np.empty(naux*nao*nmoblksize, dtype=COMPLEX)

# shranges for j3c
    mem_avail -= mem_XYblk*nmoblksize
    j3cblksize = (nkptij+nkptjmax+2)*naux # add 2 for Lpq and Lqp
    mem_j3cblk = j3cblksize*j3c_dsize/1e6
    aopblksize = min(nao*nao, mem_avail*0.7/mem_j3cblk)
    shranges = _guess_shell_ranges(mydf.cell, aopblksize, 's1')
    aopblksize = np.max([x[2] for x in shranges])
    pblksize = aopblksize // nao
    log.debug1('get_k mem_avail= %.2f MB  memj3cblk= %.2f MB', mem_avail, mem_j3cblk)
    log.debug1('get_k aopblksize= %d  pblksize= %d  nblk= %d', aopblksize, pblksize,
               len(shranges))
    log.debug1('get_k shranges= %s', shranges)
    buf_LpqR = np.empty(naux*aopblksize, dtype=REAL)
    buf_LqpR = np.empty(naux*aopblksize, dtype=REAL)
    buf_LpiR = np.empty(naux*pblksize*nmoblksize, dtype=REAL)
    buf_LpqI = np.empty(naux*aopblksize, dtype=REAL)
    buf_LqpI = np.empty(naux*aopblksize, dtype=REAL)
    buf_LpiI = np.empty(naux*pblksize*nmoblksize, dtype=REAL)

    for i0,i1 in lib.prange(0,nmomax,nmoblksize):
        di = i1 - i0
        kXipR = np.ndarray((nset,nkptij,naux,di,nao),dtype=REAL,buffer=buf_kXipR)
        kYipR = np.ndarray((nset,nkptijswap,naux,di,nao),dtype=REAL,buffer=buf_kYipR)
        kYipR.fill(0)
        kXipI = np.ndarray((nset,nkptij,naux,di,nao),dtype=REAL,buffer=buf_kXipI)
        kYipI = np.ndarray((nset,nkptijswap,naux,di,nao),dtype=REAL,buffer=buf_kYipI)
        kYipI.fill(0)

        tspans = np.zeros((9,2))
        tnames = ['buffer', 'Lpq ji', 'Lpi ji', 'kXip ji', 'Lpq ij', 'Lpi ij', 'kXip ij',
                  'j3c', 'xform']
        tick_tot = np.asarray((logger.process_clock(), logger.perf_counter()))

        p1 = 0
        for kcLpq in loop_j3c(mydf, kptij_lst=kptij_lst, aosym='s1', partition_iorj='i',
                              j3c_order='Lij', shranges=shranges, bvk_kmesh=bvk_kmesh,
                              verbose=verbose1):
            dp = kcLpq.shape[-1] // nao
            assert(dp*nao == kcLpq.shape[-1])
            p0 = p1
            p1 += dp

            tick = np.asarray((logger.process_clock(), logger.perf_counter()))

            LpqR = np.ndarray((naux,dp,nao), dtype=REAL, buffer=buf_LpqR)
            LpqI = np.ndarray((naux,dp,nao), dtype=REAL, buffer=buf_LpqI)
            LqpR = np.ndarray((naux,nao,dp), dtype=REAL, buffer=buf_LqpR)
            LqpI = np.ndarray((naux,nao,dp), dtype=REAL, buffer=buf_LqpI)
            LqiR = np.ndarray((naux*nao,di), dtype=REAL, buffer=buf_LqiR)
            LqiI = np.ndarray((naux*nao,di), dtype=REAL, buffer=buf_LqiI)
            LpiR = np.ndarray((naux*dp,di), dtype=REAL, buffer=buf_LpiR)
            LpiI = np.ndarray((naux*dp,di), dtype=REAL, buffer=buf_LpiI)

            tock = np.asarray((logger.process_clock(), logger.perf_counter()))
            tspans[0] += tock - tick

            ijswap = 0
            for kpt,adapted_kptjs,adapted_ji_idx in uniq_q_loop:
                for kptj,ji in zip(adapted_kptjs,adapted_ji_idx):
                    kj = _safe_member(kptj, kpts)
                    ki = _safe_member(kptj-kpt, kpts)
                    tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                    LpqR[:] = kcLpq[ji][0].real.reshape(naux,dp,nao)
                    LpqI[:] = kcLpq[ji][0].imag.reshape(naux,dp,nao)
                    tock = np.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[4] += tock - tick
                    for iset in range(nset):
                        moR = skmoR[iset][kj][:,i0:i1]
                        moI = skmoI[iset][kj][:,i0:i1]
                        tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                        zdotNN(LpqR.reshape(-1,nao), LpqI.reshape(-1,nao), moR, moI,
                               1, LpiR, LpiI)
                        tock = np.asarray((logger.process_clock(), logger.perf_counter()))
                        tspans[5] += tock - tick
                        kXipR[iset,ji,:,:,p0:p1] = \
                                    LpiR.reshape(naux,dp,di).transpose(0,2,1)
                        kXipI[iset,ji,:,:,p0:p1] = \
                                    LpiI.reshape(naux,dp,di).transpose(0,2,1)
                        tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                        tspans[6] += tick - tock
                    if ki != kj:
                        tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                        LqpR[:] = LpqR.transpose(0,2,1)
                        LqpI[:] = LpqI.transpose(0,2,1)
                        tock = np.asarray((logger.process_clock(), logger.perf_counter()))
                        tspans[1] += tock - tick
                        for iset in range(nset):
                            moR = skmoR[iset][ki][p0:p1,i0:i1]
                            moI = skmoI[iset][ki][p0:p1,i0:i1]
                            tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                            zdotNC(LqpR.reshape(-1,dp), LqpI.reshape(-1,dp), moR, moI,
                                   1, LqiR, LqiI)
                            tock = np.asarray((logger.process_clock(), logger.perf_counter()))
                            tspans[2] += tock - tick
                            kYipR[iset,ijswap] += \
                                        LqiR.reshape(naux,nao,di).transpose(0,2,1)
                            kYipI[iset,ijswap] += \
                                        LqiI.reshape(naux,nao,di).transpose(0,2,1)
                            tick = np.asarray((logger.process_clock(), logger.perf_counter()))
                            tspans[3] += tick - tock
                        ijswap += 1

            LpqR = LpqI = LqpR = LqpI = LqiR = LqiI = LpiR = LpiI = kcLpq = None

        tock_tot = np.asarray((logger.process_clock(), logger.perf_counter()))
        tspans[8] = tspans[:7].sum(axis=0)
        tspans[7] += tock_tot - tick_tot - tspans[8]

        for tspan,tname in zip(tspans,tnames):
            log.debug1('CPU time for get_k_kpts pass 1     %10s  %9.2f sec, '
                       'wall time  %9.2f sec', tname, *tspan)
        for tspan,tname in zip(tspans,tnames):
            if 'ij' in tname or 'ji' in tname:
                tspan_avg = tspan / max(1, nkptij if 'ji' in tname else nkptijswap)
                log.debug1('CPU time for get_k_kpts pass 1 avg %10s  %9.2f sec, '
                           'wall time  %9.2f sec', tname, *tspan_avg)

        t1 = log.timer_debug1('get_k_kpts occblk [%d:%d] pass 1'%(i0,i1), *t1)

        XiqR = np.ndarray((naux*di,nao), dtype=REAL, buffer=buf_LqiR)
        XiqI = np.ndarray((naux*di,nao), dtype=REAL, buffer=buf_LqiI)
        Xiq = np.ndarray((naux,di*nao), dtype=COMPLEX, buffer=buf_Xiq)

        kq = 0
        ijswap = 0
        for kpt,adapted_kptjs,adapted_ji_idx in uniq_q_loop:
            j2c = kj2c[kq]
            j2ctag = kj2ctag[kq]
            for kptj,ji in zip(adapted_kptjs,adapted_ji_idx):
                kj = _safe_member(kptj, kpts)
                ki = _safe_member(kptj-kpt, kpts)
                for iset in range(nset):
                    Xiq.real = kXipR[iset,ji].reshape(naux,-1)
                    Xiq.imag = kXipI[iset,ji].reshape(naux,-1)
                    if j2ctag == 'CD':
                        Xiq[:] = scipy.linalg.solve_triangular(j2c, Xiq, lower=True)
                    else:
                        Xiq[:] = lib.dot(j2c, Xiq)
                    XiqR[:] = Xiq.real.reshape(-1,nao)
                    XiqI[:] = Xiq.imag.reshape(-1,nao)
                    zdotNC(XiqR.T, XiqI.T, XiqR, XiqI, 1, vkR[iset,ki], vkI[iset,ki], 1)
                if ki != kj:
                    for iset in range(nset):
                        Xiq.real = kYipR[iset,ijswap].reshape(naux,-1)
                        Xiq.imag = kYipI[iset,ijswap].reshape(naux,-1)
                        if j2ctag == 'CD':
                            Xiq[:] = scipy.linalg.solve_triangular(j2c, Xiq, lower=True)
                        else:
                            Xiq[:] = lib.dot(j2c, Xiq)
                        XiqR[:] = Xiq.real.reshape(-1,nao)
                        XiqI[:] = Xiq.imag.reshape(-1,nao)
                        zdotCN(XiqR.T, XiqI.T, XiqR, XiqI, 1, vkR[iset,kj], vkI[iset,kj], 1)
                    ijswap += 1
            kq += 1

        XiqR = XiqI = Xiq = None

        t1 = log.timer_debug1('get_k_kpts occblk [%d:%d] pass 2'%(i0,i1), *t1)

    vk_kpts = vkR + vkI * 1j
    vk_kpts *= 1./nkpts

    return vk_kpts

def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None, exxdiv=None,
               bvk_kmesh=None):
    r'''

    dm_kpts = (nset,nkpts,nao,nao) or (nset*nkpts,nao,nao)
    dm_kpts can be tagged with mo_coeff and mo_occ
        if nset > 1:
            mo_coeff = (nset,nkpts,nao,nmo)
            mo_occ   = (nset,nkpts,nmo)
        else:
            mo_coeff = (nkpts,nao,nmo)
            mo_occ   = (nkpts,nmo)
    where the first two dims can be list or tuple.
    '''
    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    verbose1 = mydf.verbose - 2

    if exxdiv is not None and exxdiv != 'ewald':
        log.warn('RSDF does not support exxdiv %s. '
                 'exxdiv needs to be "ewald" or None', exxdiv)
        raise RuntimeError('RSGDF does not support exxdiv %s' % exxdiv)

    if kpts_band is not None:
        log.warn('RSDF get_k_kpts for band calculations is not implemented.')
        raise NotImplementedError

    kpts = lib.asarray(kpts).reshape(-1,3)
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    if mydf.auxcell is None:
        mydf.build()
    naux = mydf.auxcell.nao_nr()

    t1 = (logger.process_clock(), logger.perf_counter())

# set up mo_coeff, mo_occ
    dm_kpts_ = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts_, kpts)
    nset, nkpts, nao = dms.shape[:3]
    if getattr(dm_kpts, 'mo_coeff', None) is not None:
        mo_coeff = dm_kpts.mo_coeff
        mo_occ   = dm_kpts.mo_occ
        if nset == 1 and len(mo_coeff) == nkpts and len(mo_coeff[0]) == nao:
            mo_coeff = [mo_coeff]
            mo_occ = [mo_occ]
        elif not (len(mo_coeff) == nset and len(mo_coeff[0]) == nkpts and
                  len(mo_occ)   == nset and len(mo_occ[0])   == nkpts):
            log.error('Input mo_coeff or mo_occ has wrong dim.')
            raise ValueError
        log.debug1('Input mo_coeff and mo_occ found via tagged dm_kpts.')
    else:
        log.debug1('Diagonalizing dm_kpts to generate mo_coeff and mo_occ')
        xs = [_eigh_rdm1(dms[i]) for i in range(nset)]
        mo_coeff = [x[0] for x in xs]
        mo_occ = [x[1] for x in xs]
        xs = None

    xs = [_format_mo_coeff(mo_coeff[i], mo_occ[i], order='F') for i in range(nset)]
    skmoR = [x[0] for x in xs]
    skmoI = [x[1] for x in xs]
    xs = None
    if not any(skmoI): skmoI = None
    nmomax = np.max([moR.shape[1] for kmoR in skmoR for moR in kmoR])

    mo_isreal = skmoI is None
    j3c_isreal = gamma_point(kpts) and gamma_point(kpts_band)
    if mo_isreal and j3c_isreal and nkpts == 1: # gamma point
        smo = [kmoR[0] for kmoR in skmoR]
        vk_kpts = get_k_kpts_gamma(mydf, smo)
    else:
        if mo_isreal:
            skmoI = [[np.zeros_like(moR) for moR in kmoR] for kmoR in skmoR]
        vk_kpts = get_k_kpts_complex(mydf, skmoR, skmoI, kpts, bvk_kmesh=bvk_kmesh)

    if exxdiv == 'ewald':
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)


def _eigh_rdm1(dm_kpts, thr_nonzero=EIGH_DM_THRESH):
    assert(dm_kpts.ndim == 3)
    nkpts = len(dm_kpts)
    mo_coeff = [None] * nkpts
    mo_occ = [None] * nkpts
    for k,dm in enumerate(dm_kpts):
        e, u = scipy.linalg.eigh(dm)
        if np.any(e < -thr_nonzero):
            raise RuntimeError('Input dm is not PSD.')
        idx1 = np.where(e > thr_nonzero)[0]
        idx2 = np.asarray([i for i in range(len(e)) if i not in idx1])
        mo_occ[k] = np.concatenate([e[idx1], np.zeros_like(e[idx2])])
        mo_coeff[k] = np.hstack([u[:,idx1], u[:,idx2]])
    return mo_coeff, mo_occ
def _format_mo_coeff(mo_coeff, mo_occ, order='C'):
    nkpts = len(mo_coeff)
    nao = mo_coeff[0].shape[0]
    # padding mo_coeff using the maximum nmo of all kpts
    nmo = np.max([np.count_nonzero(occ>0) for k,occ in enumerate(mo_occ)])
    complex_mo = mo_coeff[0].dtype == COMPLEX
    kmoR = [np.zeros((nao,nmo), dtype=REAL, order=order) for k in range(nkpts)]
    if complex_mo:
        kmoI = [np.zeros((nao,nmo), dtype=REAL, order=order) for k in range(nkpts)]
    else:
        kmoI = None
    for k,occ in enumerate(mo_occ):
        mask = occ > 0
        nmok = np.count_nonzero(mask)
        mo = mo_coeff[k][:,mask]*np.sqrt(occ[mask])
        kmoR[k][:,:nmok] = mo.real
        if kmoI is not None:
            kmoI[k][:,:nmok] = mo.imag
    return kmoR, kmoI
