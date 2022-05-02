import numpy as np

from pyscf import gto as mol_gto
from pyscf.pbc import df
from pyscf.pbc.df import rsdf_helper
from pyscf.pbc.df import ft_ao
from pyscf.pbc.lib.kpts_helper import (is_zero, gamma_point, member, unique,
                                       KPT_DIFF_TOL)
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__

KINT_MAX = getattr(__config__, 'pbc_df_rsdf_direct_helper_kint_max', 30)
J3C_ORDER = getattr(__config__, 'pbc_df_rsdf_direct_helper_j3c_order', 'Lij')


''' These functions exist in previous implementations but are updated here.
'''
def get_aux_chg(auxcell, shls_slice=None):
    r""" Compute charge of the auxiliary basis, \int_Omega dr chi_P(r)

    Returns:
        The function returns a 1d numpy array of size auxcell.nao_nr().
    """
    G0 = np.zeros((1, 3))
    return ft_ao.ft_ao(auxcell, G0, shls_slice=shls_slice)[0].real


''' Needed functions
    - bvk_kmesh
    - cholesky_j2c
    - get_j2c
        - get_j2c_sr
        - get_j2c_lr
    - get_j3c
        - get_j3c_sr
        - get_j3c_lr
'''

''' To-do:
[x] separate the G=0 for get_j2c_sr
[x] make C code compute j3c in (L|ij) order
[x] make prescreening precomputeable
[x] support different j3c order
[x] support aosym='s2' for j_only mode
[ ] support separation of real and imag of j3c
[x] omega optimizer
'''

def kpts_to_kmesh(cell, kpts, kint_max=KINT_MAX):
    """ Check if kpt mesh includes the Gamma point. Generate the bvk kmesh
    only if it does.
    """
    kpts = np.reshape(kpts, (-1,3))
    nkpts = len(kpts)
    if nkpts == 1:  # single-kpt (either Gamma or shifted)
        return None

    scaled_k = cell.get_scaled_kpts(kpts).round(8)
    ksums = abs(scaled_k).sum(axis=1)
    mask = np.zeros_like(ksums, dtype=bool)
    found = False
    for kint in np.arange(1,kint_max):
        tmp = ksums * kint
        # mask = np.logical_or(mask, abs(np.round(tmp) - tmp) < 1e-6)
        mask |= abs(np.round(tmp) - tmp) < 1e-6
        if np.all(mask):
            found = True
            break
    if found:
        kmesh = (len(np.unique(scaled_k[:,0])),
                 len(np.unique(scaled_k[:,1])),
                 len(np.unique(scaled_k[:,2])))
    else:
        kmesh = None
    return kmesh

def cholesky_decomposed_metric(mydf, j2c, j2c_eig_always=None, linear_dep_threshold=None):
    import scipy.linalg

    log = logger.new_logger(mydf)

    if j2c_eig_always is None: j2c_eig_always = mydf.j2c_eig_always
    if linear_dep_threshold is None: linear_dep_threshold = mydf.linear_dep_threshold
    cell = mydf.cell

    j2c_negative = None
    try:
        if j2c_eig_always:
            raise scipy.linalg.LinAlgError
        j2c = scipy.linalg.cholesky(j2c, lower=True)
        j2ctag = 'CD'
    except scipy.linalg.LinAlgError:
        #msg =('===================================\n'
        #      'J-metric not positive definite.\n'
        #      'It is likely that mesh is not enough.\n'
        #      '===================================')
        #log.error(msg)
        #raise scipy.linalg.LinAlgError('\n'.join([str(e), msg]))
        w, v = scipy.linalg.eigh(j2c)
        ndrop = np.count_nonzero(w<linear_dep_threshold)
        if ndrop > 0:
            log.debug('DF metric linear dependency for kpt %s',
                      uniq_kptji_id)
            log.debug('cond = %.4g, drop %d bfns', w[-1]/w[0], ndrop)
        v1 = v[:,w>linear_dep_threshold].conj().T
        v1 /= np.sqrt(w[w>linear_dep_threshold]).reshape(-1,1)
        j2c = v1
        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            idx = np.where(w < -linear_dep_threshold)[0]
            if len(idx) > 0:
                j2c_negative = (v[:,idx]/np.sqrt(-w[idx])).conj().T
        w = v = None
        j2ctag = 'eig'
    return j2c, j2c_negative, j2ctag

def get_j2c_sr(mydf, auxcell=None, kpts=None, omega=None, verbose=None):
    r''' Calculate SR part of j2c for given kpts.
    '''
    if auxcell is None: auxcell = mydf.auxcell
    if kpts is None: kpts = np.zeros((1,3))
    if verbose is None: verbose = mydf.verbose
    if omega is None: omega = mydf.omega_j2c
    omega_j2c = abs(omega)
    j2c = rsdf_helper.intor_j2c(auxcell, omega_j2c, kpts=kpts, verbose=verbose)
    return j2c
def remove_j2c_sr_G0_(mydf, j2c, kpts, auxcell=None, omega=None, exxdiv=None):
    if auxcell is None: auxcell = mydf.auxcell
    if omega is None: omega = mydf.omega_j2c
    omega_j2c = abs(omega)

    if not exxdiv:
        if auxcell.dimension == 3:
            qaux = get_aux_chg(auxcell)

            qaux2 = None
            g0_j2c = np.pi/omega_j2c**2./auxcell.vol

            for k, kpt in enumerate(kpts):
                if is_zero(kpt):
                    if qaux2 is None:
                        qaux2 = np.outer(qaux,qaux)
                    j2c[k] -= qaux2 * g0_j2c
    return j2c
def get_j2c_lr(mydf, auxcell=None, kpts=None, omega=None, mesh=None, out=None,
               verbose=None):
    r''' Calculate LR part of j2c for given kpts.
    '''
    log = logger.new_logger(mydf, verbose)

    if auxcell is None: auxcell = mydf.auxcell
    if kpts is None: kpts = np.zeros((1,3))
    omega_j2c = abs(mydf.omega_j2c) if omega is None else abs(omega)
    mesh_j2c = mydf.mesh_j2c if mesh is None else mesh

    nkpts = len(kpts)
    naoaux = auxcell.nao_nr()
    if out is None:
        out = [np.zeros((naoaux,naoaux), dtype=np.float64) if gamma_point(kpt) else
               np.zeros((naoaux,naoaux), dtype=np.complex128) for kpt in kpts]
    j2c = out

    Gv, Gvbase, kws = auxcell.get_Gv_weights(mesh_j2c)
    b = auxcell.reciprocal_vectors()
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]

    max_memory = max(2000, mydf.max_memory - lib.current_memory()[0])
    blksize = max(2048, int(max_memory*.5e6/16/auxcell.nao_nr()))
    log.debug2('j2c_lr: max_memory %s (MB)  blocksize %s', max_memory, blksize)

    for k, kpt in enumerate(kpts):
        coulG_lr = mydf.weighted_coulG(omega_j2c, kpt, False, mesh_j2c)
        for p0, p1 in lib.prange(0, ngrids, blksize):
            aoaux = ft_ao.ft_ao(auxcell, Gv[p0:p1], None, b, gxyz[p0:p1],
                                Gvbase, kpt).T
            LkR = np.asarray(aoaux.real, order='C')
            LkI = np.asarray(aoaux.imag, order='C')
            aoaux = None

            if is_zero(kpt):  # kpti == kptj
                j2c[k] += lib.ddot(LkR*coulG_lr[p0:p1], LkR.T)
                j2c[k] += lib.ddot(LkI*coulG_lr[p0:p1], LkI.T)
            else:
                j2cR, j2cI = df.df_jk.zdotCN(LkR*coulG_lr[p0:p1],
                                             LkI*coulG_lr[p0:p1], LkR.T, LkI.T)
                j2c[k] += j2cR + j2cI * 1j

            LkR = LkI = None

    return out
def get_j2c(mydf, auxcell=None, kpts=None, omega=None, mesh=None, exxdiv=None,
            verbose=None):
    if auxcell is None: auxcell = mydf.auxcell
    if kpts is None: kpts = np.zeros((1,3))
    if verbose is None: verbose = mydf.verbose
    if omega is None: omega = mydf.omega_j2c
    omega_j2c = abs(omega)

    j2c = get_j2c_sr(mydf, auxcell=auxcell, kpts=kpts, omega=omega, verbose=verbose)
    j2c = remove_j2c_sr_G0_(mydf, j2c, kpts=kpts, auxcell=auxcell, omega=omega,
                            exxdiv=exxdiv)
    j2c = get_j2c_lr(mydf, auxcell=auxcell, kpts=kpts, omega=omega, mesh=mesh, out=j2c,
                     verbose=verbose)
    return j2c


from pyscf.pbc.df.rsdf_helper import (_get_refuniq_map,BOHR,_get_schwartz_data,
                                      _get_schwartz_dcut,_make_dijs_lst,
                                      _get_3c2e_Rcuts,_get_atom_Rcuts_3c,
                                      _get_Lsmin,KPT_DIFF_TOL,gamma_point,
                                      wrap_int3c_nospltbas)
def get_prescreening_data(cell, auxcell, omega, precision=None, estimator='ME',
                          verbose=None):
    log = logger.new_logger(cell, verbose)

# prescreening data
    t1 = (logger.process_clock(), logger.perf_counter())
    if precision is None: precision = cell.precision
    refuniqshl_map, uniq_atms, uniq_bas, uniq_bas_loc = _get_refuniq_map(cell)
    auxuniqshl_map, uniq_atms, uniq_basaux, uniq_basaux_loc = \
                                                        _get_refuniq_map(auxcell)

    dstep = 1 # 1 Ang bin size for shl pair
    dstep_BOHR = dstep / BOHR
    Qauxs = _get_schwartz_data(uniq_basaux, omega, keep1ctr=False, safe=True)
    dcuts = _get_schwartz_dcut(uniq_bas, omega, precision/Qauxs.max(),
                               r0=cell.rcut)
    dijs_lst = _make_dijs_lst(dcuts, dstep_BOHR)
    dijs_loc = np.cumsum([0]+[len(dijs) for dijs in dijs_lst]).astype(np.int32)
    if estimator.upper() in ["ISFQ0","ISFQL"]:
        Qs_lst = _get_schwartz_data(uniq_bas, omega, dijs_lst, keep1ctr=True,
                                    safe=True)
    else:
        Qs_lst = [np.zeros_like(dijs) for dijs in dijs_lst]
    Rcuts = _get_3c2e_Rcuts(uniq_bas, uniq_basaux, dijs_lst, omega, precision,
                            estimator, Qs_lst)
    bas_exps = np.array([np.asarray(b[1:])[:,0].min() for b in uniq_bas])
    atom_Rcuts = _get_atom_Rcuts_3c(Rcuts, dijs_lst, bas_exps, uniq_bas_loc,
                                    uniq_basaux_loc)
    cell_rcut = atom_Rcuts.max()
    uniqexp = np.array([np.asarray(b[1:])[:,0].min() for b in uniq_bas])
    nbasauxuniq = len(uniq_basaux)
    dcut2s = dcuts**2.
    Rcut2s = Rcuts**2.
    Ls = _get_Lsmin(cell, atom_Rcuts, uniq_atms)
    prescreening_data = (refuniqshl_map, auxuniqshl_map, nbasauxuniq, uniqexp,
                         dcut2s, dstep_BOHR, Rcut2s, dijs_loc, Ls)
    log.debug("j3c prescreening: cell rcut %.2f Bohr  keep %d imgs",
              cell_rcut, Ls.shape[0])
    t1 = log.timer_debug1('prescrn warmup', *t1)
    return prescreening_data
def get_int3c(cell, auxcell, omega, precision=None, kptij_lst=np.zeros((1,2,3)),
              intor='int3c2e', comp=None, estimator='ME', verbose=None,
              bvk_kmesh=None, aosym='s2ij', j3c_order=J3C_ORDER):
    prescreening_data = get_prescreening_data(cell, auxcell, omega,
                                              precision=precision,
                                              estimator=estimator,
                                              verbose=verbose)

    intor, comp = mol_gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)

    shlpr_mask = np.ones((cell.nbas, cell.nbas), dtype=np.int8, order="C")

    int3c = wrap_int3c_nospltbas(cell, auxcell, omega, shlpr_mask,
                                 prescreening_data, intor, aosym, comp,
                                 kptij_lst, bvk_kmesh=bvk_kmesh, order=j3c_order,
                                 verbose=verbose)
    return int3c
def aux_e2_nospltbas(cell, auxcell_or_auxbasis, omega, intor='int3c2e', aosym='s2ij',
                     j3c_order=J3C_ORDER, comp=None, kptij_lst=np.zeros((1,2,3)),
                     shls_slice=None, bvk_kmesh=None, precision=None, estimator="ME",
                     int3c=None, verbose=None, out=None):
    r'''3-center AO integrals (ij|L) with double lattice sum:
    \sum_{lm} (i[l]j[m]|L[0]), where L is the auxiliary basis.
    Three-index integral tensor (kptij_idx, nao_pair, naux) or four-index
    integral tensor (kptij_idx, comp, nao_pair, naux) are held in memory
    (which is the main difference from :func:"_aux_e2_nospltbas").

    **This function should be only used by RSGDF**

    Args:
        shls_slice (list/tuple):
            (ish0, ish1, jsh0, jsh1, ksh0, ksh1)
            where i and j are AO indices and k is aux index.
            If aosym = "s2", jsh0 = 0 and jsh1 = ish1 must be used.
        kptij_lst : (*,2,3) array
            A list of (kpti, kptj)
        int3c (callable):
            Generated by wrap_int3c_nospltbas. This argument allows using pre-generation
            of int3c.
        out (numpy array; default None):
            If provided, the output array will be constructed using this buffer.
    Returns:
        A numpy array of shape (nkpts,comp,naopair,naux), where
            naopair = ni*(ni+1)//2 if j_only = True else ni*nj
    '''
    if verbose is None: verbose = cell.verbose
    log = logger.Logger(cell.stdout, verbose)

    if isinstance(auxcell_or_auxbasis, mol_gto.Mole):
        auxcell = auxcell_or_auxbasis
    else:
        from pyscf.pbc.df.incore import make_auxcell
        auxcell = make_auxcell(cell, auxcell_or_auxbasis)

    if not callable(int3c):
        int3c = get_int3c(cell, auxcell, omega, precision=precision, kptij_lst=kptij_lst,
                      intor=intor, comp=comp, estimator=estimator, verbose=verbose,
                      bvk_kmesh=bvk_kmesh, aosym=aosym, j3c_order=j3c_order)

    intor, comp = mol_gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)

    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)

    ao_loc = cell.ao_loc_nr()
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)[:shls_slice[5]+1]
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    nk = aux_loc[shls_slice[5]] - aux_loc[shls_slice[4]]
    nkptij = len(kptij_lst)

    nii = (ao_loc[shls_slice[1]]*(ao_loc[shls_slice[1]]+1)//2 -
           ao_loc[shls_slice[0]]*(ao_loc[shls_slice[0]]+1)//2)
    nij = ni * nj

    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]
    aosym_ks2 = abs(kpti-kptj).sum(axis=1) < KPT_DIFF_TOL
    j_only = np.all(aosym_ks2)
    aosym_ks2 &= aosym[:2] == 's2'

    if j_only and aosym[:2] == 's2':
        assert(shls_slice[2] == 0)
        nao_pair = nii
    else:
        nao_pair = nij

    if gamma_point(kptij_lst):
        dtype = np.double
        dsize = 8
    else:
        dtype = np.complex128
        dsize = 16

    if j3c_order == 'Lij':
        bufshape = (nkptij,comp,nk,nao_pair)
    else:
        bufshape = (nkptij,comp,nao_pair,nk)
    bufsize = np.prod(bufshape)
    if out is None:
        bufmem = bufsize * dsize / 1e6
        log.debug1("allocating %s MB of mem", bufmem)
        out = np.empty(bufsize, dtype=dtype)
    elif isinstance(out, np.ndarray):
        if out.nbytes < bufsize*dsize:
            raise RuntimeError("Input buffer size %d is too small (need %d).",
                               out.nbytes//dsize, bufsize)
    else:
        raise TypeError('out must be None or numpy array.')
    buf = np.ndarray(bufshape, dtype=dtype, buffer=out)

    int3c(shls_slice, buf)

    return buf
def get_j3c_sr(mydf, cell=None, auxcell=None, kptij_lst=None, shls_slice=None,
               omega=None, aosym='s2ij', j3c_order=J3C_ORDER, comp=None,
               bvk_kmesh=None, precision=None,
               estimator='ME', int3c=None, verbose=None, out=None):
    r''' Compute the SR part of the j3c tensor. Shape = (nkptij,comp,naoaux,nao_pair)
    '''
    if cell is None: cell = mydf.cell
    if auxcell is None: auxcell = mydf.auxcell
    if omega is None: omega = mydf.omega
    if precision is None: precision = mydf.precision_R
    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)
    if verbose is None: verbose = mydf.verbose
    # out.shape = (nkptij, comp, nao_pair, naoaux)
    out = aux_e2_nospltbas(cell, auxcell, omega, intor='int3c2e',
                           aosym=aosym, j3c_order=j3c_order,
                           comp=comp, kptij_lst=kptij_lst,
                           shls_slice=shls_slice, bvk_kmesh=bvk_kmesh,
                           precision=precision, estimator=estimator,
                           int3c=int3c, verbose=verbose, out=out)
    return out
def remove_j3c_sr_G0_q_(mydf, j3c, shls_slice, kptij_lst, cell=None, auxcell=None,
                        omega=None, aosym='s2ij', j3c_order=J3C_ORDER, exxdiv=None,
                        verbose=None):
    r''' Calculate G=0 correction to SR j3c. This will modify the input j3c in situ.

    Args:
        aosym (str):
            's1' or 's2'. This has to be consistent with the input j3c, shls_slice,
            and kptij_lst.
    '''
    if cell is None: cell = mydf.cell
    if auxcell is None: auxcell = mydf.auxcell
    if omega is None: omega = mydf.omega

    log = logger.new_logger(mydf, verbose)

    if not exxdiv:
        if cell.dimension == 3:
            ao_loc = cell.ao_loc_nr()
            ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
            nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
            nii_start = ao_loc[shls_slice[0]]*(ao_loc[shls_slice[0]]+1)//2
            nii_end = ao_loc[shls_slice[1]]*(ao_loc[shls_slice[1]]+1)//2
            nii = nii_end - nii_start
            nij = ni * nj

            if aosym[:2] == 's2':
                # check if aosym is consistent with kptij_lst and shls_slice
                assert(shls_slice[2] == 0)
                assert(is_j_only(kptij_lst))
                nao_pair = nii
            else:
                nao_pair = nij

            # check if aosym is consistent with j3c shape
            if j3c_order == 'Lij':
                nao_pair_j3c = j3c.shape[-1]
            else:
                nao_pair_j3c = j3c.shape[-2]
            assert(nao_pair_j3c == nao_pair)

            g0 = np.pi/omega**2./cell.vol
            qaux = get_aux_chg(auxcell, shls_slice=shls_slice[-2:])
            verbose_loop = mydf.verbose - 2 # print only if verbose>=8
            for q, adapted_kptjs, adapted_ji_idx in loop_uniq_q(mydf,
                                                                kptij_lst=kptij_lst,
                                                                verbose=verbose_loop):
                if not is_zero(q):
                    continue

                # TODO: calculate ovlp for shls_slice only
                # @@HY: no need. FWIW this function seems to NEVER take >1% of the time

                vbar = qaux * g0
                ovlp = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=adapted_kptjs)

                if aosym[:2] == 's2':
                    ovlp = [lib.pack_tril(s)[nii_start:nii_end] for s in ovlp]
                else:
                    ovlp = [s[ao_loc[shls_slice[0]]:ao_loc[shls_slice[1]],
                              ao_loc[shls_slice[2]]:ao_loc[shls_slice[3]]].reshape(-1)
                            for s in ovlp]

                if j3c_order == 'Lij':
                    for k, idx in enumerate(adapted_ji_idx):
                        for i in np.where(vbar != 0)[0]:
                            j3c[idx,0,i] -= vbar[i] * ovlp[k]
                else:
                    for k, idx in enumerate(adapted_ji_idx):
                        for i in np.where(vbar != 0)[0]:
                            j3c[idx,0,:,i] -= vbar[i] * ovlp[k]
    else:
        raise NotImplementedError
    return j3c
def add_j3c_lr_q_(mydf, j3c, kpt, adapted_kptjs, adapted_ji_idx,
                  cell=None, auxcell=None, shls_slice=None, omega=None, mesh=None,
                  aosym='s2ij', j3c_order=J3C_ORDER, comp=None, gLRI=None,
                  bvk_kmesh=None, verbose=None):
    r''' Add LR part of j3c to input j3c
    '''
    log = logger.new_logger(mydf, verbose)

    if comp is None: comp = 1
    if comp != 1:
        raise NotImplementedError

    if cell is None: cell = mydf.cell
    if auxcell is None: auxcell = mydf.auxcell
    if omega is None: omega = mydf.omega
    if mesh is None: mesh = mydf.mesh_compact
    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)

    nkptj = len(adapted_kptjs)

# determine nao_pair
    ao_loc = cell.ao_loc_nr()
    aux_loc = auxcell.ao_loc_nr()
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    naoaux = aux_loc[shls_slice[5]] - aux_loc[shls_slice[4]]
    nii_start = ao_loc[shls_slice[0]]*(ao_loc[shls_slice[0]]+1)//2
    nii_end = ao_loc[shls_slice[1]]*(ao_loc[shls_slice[1]]+1)//2
    nii = nii_end - nii_start
    nij = ni * nj

    if is_zero(kpt):
        aosym_ = aosym[:2]
    else:
        aosym_ = 's1'

    ncol = nii if aosym_ == 's2' else nij

# check j3c dtype and shape
    if j3c.dtype == np.double and not (is_zero(kpt) and is_zero(adapted_kptjs)):
        log.error('input j3c is real but input kpt/adapted_kptjs are not gamma point')
        raise ValueError

    j3c_shape = (naoaux,ncol) if j3c_order == 'Lij' else (ncol,naoaux)
    if j3c.shape[-2:] != j3c_shape:
        log.error('Input j3c has a wrong shape. Expecting j3c.shape[-2:]= %s, '
                  'getting %s.', j3c_shape, j3c.shape[-2:])
        raise ValueError

# useful constants
    nbas = cell.nbas
    nao = cell.nao_nr()
    nbasaux = auxcell.nbas
    naoaux = auxcell.nao_nr()
    nkptj = len(adapted_kptjs)

    mesh = mydf.mesh_compact
    b = cell.reciprocal_vectors()
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]

# bra
    if gLRI is None:
        auxshls_slice = (shls_slice[-2], shls_slice[-1])
        Gaux = ft_ao.ft_ao(auxcell, Gv, auxshls_slice, b, gxyz, Gvbase, kpt)
        wcoulG_lr = mydf.weighted_coulG(omega, kpt, False, mesh)
        Gaux *= wcoulG_lr.reshape(-1,1)
        gLR = Gaux.real.copy('C')
        gLI = Gaux.imag.copy('C')
        Gaux = None
    else:
        gLR, gLI = gLRI

# set up buffer
    # buffer for out
    kLpqRbuf = np.zeros((nkptj,*j3c_shape), dtype=np.double)
    if j3c[0].dtype == np.complex128:
        kLpqIbuf = np.zeros((nkptj,*j3c_shape), dtype=np.double)
    # (pq|G;{kj}) + (pq|G) ==> ncol*(nkptj+1)*Gblksize
    mem_avail = mydf.max_memory - lib.current_memory()[0]
    Gblksize = max(16, int(np.floor(mem_avail*0.4 / (ncol*(nkptj+1)*16/1e6))))
    Gblksize = min(Gblksize, ngrids, 16384)
    pqgRbuf = np.empty(Gblksize*ncol, dtype=np.double)
    pqgIbuf = np.empty(Gblksize*ncol, dtype=np.double)
    buf = np.empty(nkptj*Gblksize*ncol, dtype=np.complex128)
    for p0, p1 in lib.prange(0, ngrids, Gblksize):
        # shape: nkptj, nG, ncol
        dat = ft_ao.ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym_,
                                   b, gxyz[p0:p1], Gvbase, kpt,
                                   adapted_kptjs, out=buf,
                                   bvk_kmesh=bvk_kmesh)
        nG = p1 - p0
        for k, ji in enumerate(adapted_ji_idx):
            aoao = dat[k].reshape(nG,ncol)
            pqgR = np.ndarray((ncol,nG), buffer=pqgRbuf)
            pqgI = np.ndarray((ncol,nG), buffer=pqgIbuf)
            pqgR[:] = aoao.real.T
            pqgI[:] = aoao.imag.T

            if j3c_order == 'Lij':
                j3c_kR = kLpqRbuf[k]
                lib.dot(gLR[p0:p1].T, pqgR.T, 1, j3c_kR, 1)
                lib.dot(gLI[p0:p1].T, pqgI.T, 1, j3c_kR, 1)
                if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                    j3c_kI = kLpqIbuf[k]
                    lib.dot(gLR[p0:p1].T, pqgI.T, 1, j3c_kI, 1)
                    lib.dot(gLI[p0:p1].T, pqgR.T, -1, j3c_kI, 1)
            else:
                j3c_kR = kLpqRbuf[k]
                lib.dot(pqgR, gLR[p0:p1], 1, j3c_kR, 1)
                lib.dot(pqgI, gLI[p0:p1], 1, j3c_kR, 1)
                if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                    j3c_kI = kLpqIbuf[k]
                    lib.dot(pqgI, gLR[p0:p1], 1, j3c_kI, 1)
                    lib.dot(pqgR, gLI[p0:p1], -1, j3c_kI, 1)

    for k, ji in enumerate(adapted_ji_idx):
        if j3c[ji].dtype == np.complex128:
            j3c[ji][0] += kLpqRbuf[k] + kLpqIbuf[k] * 1j
        else:
            j3c[ji][0] += kLpqRbuf[k]

    return j3c
def add_j3c_lr_(mydf, j3c, cell=None, auxcell=None, kptij_lst=None, shls_slice=None,
                omega=None, mesh=None, aosym='s2ij', j3c_order=J3C_ORDER, comp=None,
                bvk_kmesh=None, kgLRI=None, verbose=None):
    r''' Add the LR part of j3c to input j3c.
    '''
    log = logger.new_logger(mydf, verbose)

    if cell is None: cell = mydf.cell
    if auxcell is None: auxcell = mydf.auxcell
    if omega is None: omega = mydf.omega
    if mesh is None: mesh = mydf.mesh_compact
    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)

    if aosym[:2] == 's2' and is_j_only(kptij_lst) and shls_slice[2] == 0:
        aosym = 's2'
    else:
        aosym = 's1'

    if kptij_lst is None: kptij_lst = np.zeros((1,2,3))

    verbose_loop = mydf.verbose - 2 # print only if verbose>=8
    kq = 0
    for kpt,adapted_kptjs,adapted_ji_idx in loop_uniq_q(mydf, kptij_lst=kptij_lst,
                                                        verbose=verbose_loop):
        gLRI = kgLRI[kq] if kgLRI is not None else None
        add_j3c_lr_q_(mydf, j3c, kpt, adapted_kptjs, adapted_ji_idx,
                      cell=cell, auxcell=auxcell, shls_slice=shls_slice,
                      omega=omega, mesh=mesh, aosym=aosym, j3c_order=j3c_order,
                      comp=comp, bvk_kmesh=bvk_kmesh, verbose=verbose, gLRI=gLRI)
        kq += 1
    return j3c
def get_j3c(mydf, cell=None, auxcell=None, kptij_lst=None, shls_slice=None,
            omega=None, aosym='s2ij', j3c_order=J3C_ORDER, comp=None,
            bvk_kmesh_R=None, bvk_kmesh_G=None,
            precision=None, mesh=None, estimator='ME', exxdiv=None,
            int3c=None, kgLRI=None, out=None, verbose=None):
    if cell is None: cell = mydf.cell
    if auxcell is None: auxcell = mydf.auxcell
    if kptij_lst is None: np.zeros((1,2,3)) # gamma point only
    if shls_slice is None: shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)
    if omega is None: omega = mydf.omega
    if precision is None: precision = mydf.precision_R
    if mesh is None: mesh = mydf.mesh_compact
    if verbose is None: verbose = mydf.verbose

    # determine aosym
    if aosym[:2] == 's2' and is_j_only(kptij_lst) and shls_slice[2] == 0:
        aosym = 's2'
    else:
        aosym = 's1'

    log = logger.new_logger(mydf, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    j3c = get_j3c_sr(mydf, cell=cell, auxcell=auxcell, kptij_lst=kptij_lst,
                     shls_slice=shls_slice, omega=omega, aosym=aosym,
                     j3c_order=j3c_order, comp=comp, bvk_kmesh=bvk_kmesh_R,
                     precision=precision, estimator=estimator,
                     int3c=int3c, out=out, verbose=verbose)
    t1 = log.timer_debug1('j3c_sr', *t0)
    remove_j3c_sr_G0_q_(mydf, j3c, shls_slice, kptij_lst, cell=cell, auxcell=auxcell,
                        omega=omega, aosym=aosym, j3c_order=j3c_order, exxdiv=exxdiv,
                        verbose=verbose)
    t1 = log.timer_debug1('j3c_g0', *t1)
    add_j3c_lr_(mydf, j3c, cell=cell, auxcell=auxcell, kptij_lst=kptij_lst,
                shls_slice=shls_slice, omega=omega, mesh=mesh, aosym=aosym,
                j3c_order=j3c_order, comp=comp, bvk_kmesh=bvk_kmesh_G,
                kgLRI=kgLRI, verbose=verbose)
    t1 = log.timer_debug1('j3c_lr', *t1)
    return j3c

def loop_j3c(mydf, kptij_lst=np.zeros((1,2,3)), aosym='s1', j3c_order=J3C_ORDER,
             partition_iorj='i', max_memory=2000, blksize=None, shranges=None,
             verbose=None, **kwargs):
    r''' Return j3c in (L|ij) form where the AO pair index 'ij' is partitioned.

    Args:
        kptij_lst (np.ndarray):
            kptis = kptij_lst[:,0]
            kptjs = kptij_lst[:,1]
        aosym (str):
            Symmetry of the AO pair, can be 's1' or 's2'.
            Currently only 's1' is supported.
        j3c_order (str):
            How are the j3c integrals arranged?
                - 'ijL': return j3c tensor in the form (ij|L)
                - 'Lij': return j3c tensor in the form (L|ij)
        partition_iorj (str):
            Which of the two AO indices is partitioned, can be 'i' or 'j'.
            For j3c_order = 'Lij':
                - 'i': return (L|[i0:i1]j), (L|[i1:i2]j), ...
                - 'j': return (L|i[j0:j1]), (L|i[j1:j2]), ...
            For j3c_order = 'ijL':
                - 'i': return ([i0:i1]j|L), ([i1:i2]j|L), ...
                - 'j': return (i[j0:j1]|L), (i[j1:j2]|L), ...
        max_memory (float) | blksize (int) | shranges (array-like, shape (x,3)):
            These arguments are related and together determine the AO pair blocks.
            - 'shranges' has the format
                [(shl0,shl1,ncol01), (shl1,shl2,ncol12), ...]
            which means that the i or j index is split into blocks given by shl0, shl1,
            shl2, etc. and ncol01, ncol02, etc. specify the size of the corresponding
            AO pair block. If 'shranges' is provided, the other two are ignored.
            - 'blksize' is the maximum AO pair block size, which will be used to
            generate 'shranges' if the latter is not provided. If provided, 'max_memory'
            is ignored.
            - 'max_memory' is the maximum allowed memory for each AO pair block. If
            neither 'shranges' nor 'blksize' is provided, 'blksize' is deduced from
            'max_memory'.
        verbose:
            Control verbose level. Default is mydf.verbose

    Kwargs:
        bvk_kmesh | bvk_kmesh_R | bvk_kmesh_G (array-like):
            bvk_kmesh_R/G is the bvk_kmesh for SR/LR j3c. If either is not provided, its
            value is set to bvk_kmesh. Default is None for all three, i.e., not use bvk.
    '''
    from pyscf.df.outcore import _guess_shell_ranges

    log = logger.new_logger(mydf, verbose=verbose)

    # sanity check
    valueerror = False
    if aosym[:2] not in ['s1','s2']:
        log.error('Invalid aosym %s (must start with "s1" or "s2").', aosym)
        valueerror = True
    if partition_iorj not in 'ij':
        log.error('Invalid partition_iorj %s (must be "i" or "j").', partition_iorj)
        valueerror = True
    if valueerror:
        raise ValueError

    # kwargs:
    bvk_kmesh = kwargs.get('bvk_kmesh', None)
    if hasattr(bvk_kmesh, '__len__') and len(bvk_kmesh) == 2:
        bvk_kmesh_R , bvk_kmesh_G = bvk_kmesh
    else:
        bvk_kmesh_R = bvk_kmesh_G = bvk_kmesh
    log.debug1('Using bvk_kmesh_R= %s  bvk_kmesh_G= %s', bvk_kmesh_R, bvk_kmesh_G)

    cell = mydf.cell
    auxcell = mydf.auxcell
    nao = cell.nao_nr()
    naoaux = auxcell.nao_nr()

    nkptij = len(kptij_lst)
    xs = [x for x in loop_uniq_q(mydf, kptij_lst=kptij_lst, verbose=0)]
    nkptjmax = np.max([len(x[1]) for x in xs])
    uniq_kpts = np.asarray([x[0] for x in xs])
    xs = None

    rowlen = (nkptij + nkptjmax) * naoaux
    if aosym[:2] == 's2':
        if not is_j_only(kptij_lst):
            log.error('aosym = "s2" must be used with kpti = kptj, i.e., j-only mode.')
            raise RuntimeError

        if j3c_order == 'Lij':
            log.error('s2 symmetry for j3c_order = "Lij" is not implemented (yet). '
                      'Use j3c_order = "ijL" instead.')
            raise NotImplementedError

        nao_pair = nao*(nao+1)//2
        if partition_iorj == 'i':
            get_shls_slice = lambda p0,p1: (p0,p1,0,p1,0,auxcell.nbas)
        else:
            log.error('aosym = "s2" must be used with partition_iorj = "i".')
            raise ValueError
    else:
        nao_pair = nao*nao
        if partition_iorj == 'i':
            get_shls_slice = lambda p0,p1: (p0,p1,0,cell.nbas,0,auxcell.nbas)
        elif partition_iorj == 'j':
            get_shls_slice = lambda p0,p1: (0,cell.nbas,p0,p1,0,auxcell.nbas)

    if is_zero(kptij_lst):  # all gamma point
        dtype = np.float64
        dsize = 8
    else:
        dtype = np.complex128
        dsize = 16

    if shranges is None:
        if blksize is None:
            info_blksize = ' deduced from max_memory= %.1f' % max_memory
            blksize = min(nao_pair, int(np.floor(max_memory*1e6 / (rowlen * dsize))))
        else:
            info_blksize = ''
        shranges = _guess_shell_ranges(mydf.cell, blksize, aosym)
        blksizemax = max([x[-1] for x in shranges])
        if blksizemax > blksize:
            log.warn('In loop_j3c: actual block size= %d is greater than input= %d%s.',
                     blksizemax, blksize, info_blksize)
    else:
        blksizemax = max([x[-1] for x in shranges])
    buf = np.empty(rowlen*blksizemax, dtype=dtype)

    if len(shranges) > 1:
        # precompute int3c
        int3c = get_int3c(cell, auxcell, mydf.omega, precision=mydf.precision_R,
                          kptij_lst=kptij_lst, verbose=log.verbose, bvk_kmesh=bvk_kmesh_R,
                          aosym=aosym, j3c_order=j3c_order)
        # precompute kgLR/I
        mesh = mydf.mesh_compact
        omega = mydf.omega
        b = cell.reciprocal_vectors()
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        ngrids = gxyz.shape[0]
        auxshls_slice = (0, auxcell.nbas)
        kgLRI = np.empty((len(uniq_kpts),2,ngrids,naoaux), dtype=np.float64)
        for k,kpt in enumerate(uniq_kpts):
            Gaux = ft_ao.ft_ao(auxcell, Gv, auxshls_slice, b, gxyz, Gvbase, kpt)
            wcoulG_lr = mydf.weighted_coulG(omega, kpt, False, mesh)
            Gaux *= wcoulG_lr.reshape(-1,1)
            kgLRI[k,0] = Gaux.real
            kgLRI[k,1] = Gaux.imag
            Gaux = None
    else:
        int3c = kgLRI = None

    p1 = 0
    for ipart,shrange in enumerate(shranges):
        t1 = (logger.process_clock(), logger.perf_counter())

        s0, s1, ncol = shrange
        p0 = p1
        p1 = p0 + ncol

        shls_slice = get_shls_slice(s0,s1)
        j3c = get_j3c(mydf, kptij_lst=kptij_lst, shls_slice=shls_slice, aosym=aosym,
                      j3c_order=j3c_order, out=buf,
                      bvk_kmesh_R=bvk_kmesh_R, bvk_kmesh_G=bvk_kmesh_G,
                      verbose=log.verbose, int3c=int3c, kgLRI=kgLRI)

        t1 = log.timer('j3c [%d:%d]'%(p0,p1), *t1)

        yield j3c
        j3c = None

def search_best_omega(mydf, omega_mesh=None, nsample=3, kptij_lst=np.zeros((1,2,3)),
                      verbose=None):
    log = logger.new_logger(mydf, verbose)

    if omega_mesh is None: omega_mesh = np.arange(0.1,0.41,0.1)
    log.debug('Searching best omega from %s', omega_mesh)
    n = len(omega_mesh)

    aosym = 's1'
    j3c_order = 'Lij'
    cell = mydf.cell
    auxcell = mydf.auxcell
    nao = cell.nao_nr()
    naoaux = auxcell.nao_nr()

    nkptij = len(kptij_lst)
    xs = [x for x in loop_uniq_q(mydf, kptij_lst=kptij_lst, verbose=0)]
    uniq_kpts = np.asarray([x[0] for x in xs])
    xs = None

    if isinstance(mydf.use_bvk, bool):
        use_bvk_R = use_bvk_G = mydf.use_bvk
    else:
        use_bvk_R,  use_bvk_G = mydf.use_bvk
    if use_bvk_R or use_bvk_G:
        from pyscf.pbc.df.rsdf_direct_helper import kpts_to_kmesh
        bvk_kmesh0 = kpts_to_kmesh(cell, mydf.kpts)
        bvk_kmesh = [bvk_kmesh0 if use_bvk_R else None,
                     bvk_kmesh0 if use_bvk_G else None]
    else:
        bvk_kmesh = None
    if hasattr(bvk_kmesh, '__len__') and len(bvk_kmesh) == 2:
        bvk_kmesh_R , bvk_kmesh_G = bvk_kmesh
    else:
        bvk_kmesh_R = bvk_kmesh_G = bvk_kmesh

# determine shls_slice
    nbas = cell.nbas
    nbasaux = auxcell.nbas
    mauxshl = 10
    ishls = np.random.randint(nbas, size=nsample)
    jshls = np.random.randint(nbas, size=nsample)
    kshls = np.random.randint(nbasaux-mauxshl, size=nsample)
    shls_slice_list = [(ishl,ishl+1,jshl,jshl+1,kshl,kshl+mauxshl)
                       for ishl,jshl,kshl in zip(ishls,jshls,kshls)]
    log.debug('Using shls_slice= %s for timing', shls_slice_list)

    dts = np.zeros((n,2))
    for i,omega in enumerate(omega_mesh):
        with lib.temporary_env(mydf.cell, verbose=0):
            mydf_ = df.RSDF(mydf.cell, mydf.kpts).set(direct=mydf.direct,
                                                      use_bvk=mydf.use_bvk,
                                                      precision_R=mydf.precision_R,
                                                      precision_G=mydf.precision_G,
                                                      omega=omega, verbose=0,
                                                      auxbasis=mydf.auxcell._basis)
            mydf_.build()
        # precompute int3c
        int3c = get_int3c(cell, auxcell, mydf_.omega,
                          precision=mydf_.precision_R,
                          kptij_lst=kptij_lst, verbose=0,
                          bvk_kmesh=bvk_kmesh_R,
                          aosym=aosym, j3c_order=j3c_order)
        # precompute kgLR/I
        mesh = mydf_.mesh_compact
        b = cell.reciprocal_vectors()
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        ngrids = gxyz.shape[0]
        aux_loc = auxcell.ao_loc_nr()
        # timing ints
        for shls_slice in shls_slice_list:
            # precompute kgLR/I
            auxshls_slice = shls_slice[-2:]
            naoaux_blk = aux_loc[auxshls_slice[1]] - aux_loc[auxshls_slice[0]]
            kgLRI = np.empty((len(uniq_kpts),2,ngrids,naoaux_blk), dtype=np.float64)
            for k,kpt in enumerate(uniq_kpts):
                Gaux = ft_ao.ft_ao(auxcell, Gv, auxshls_slice, b, gxyz, Gvbase, kpt)
                wcoulG_lr = mydf_.weighted_coulG(omega, kpt, False, mesh)
                Gaux *= wcoulG_lr.reshape(-1,1)
                kgLRI[k,0] = Gaux.real
                kgLRI[k,1] = Gaux.imag
                Gaux = None
            t0 = np.asarray((logger.process_clock(), logger.perf_counter()))
            get_j3c(mydf_, kptij_lst=kptij_lst, shls_slice=shls_slice,
                    aosym=aosym, j3c_order=j3c_order, bvk_kmesh_R=bvk_kmesh_R,
                    bvk_kmesh_G=bvk_kmesh_G, verbose=0, int3c=int3c, kgLRI=kgLRI)
            t1 = np.asarray((logger.process_clock(), logger.perf_counter()))
            dts[i] += t1 - t0
        log.debug('omega= %5.2f   tcpu= %9.6f sec  twall= %9.6f sec', omega, *dts[i])
    imin = np.argmin(dts[:,1])
    omega = omega_mesh[imin]
    log.info('Optimal omega= %5.2f', omega)

    return omega


def get_kptij_lst(kpts, kpts_band=None, j_only=False, ksym='s2'):
    uniq_idx = unique(kpts)[1]
    kpts = np.asarray(kpts)[uniq_idx]
    if kpts_band is None:
        kband_uniq = np.zeros((0,3))
    else:
        kband_uniq = [k for k in kpts_band if len(member(k, kpts))==0]
    if j_only:
        kall = np.vstack([kpts,kband_uniq])
        kptij_lst = np.hstack((kall,kall)).reshape(-1,2,3)
    else:
        if ksym == 's2':
            kptij_lst = [(ki, kpts[j]) for i, ki in enumerate(kpts) for j in range(i+1)]
        else:
            kptij_lst = [(ki, kj) for ki in kpts for kj in kpts]
        kptij_lst.extend([(ki, kj) for ki in kband_uniq for kj in kpts])
        kptij_lst.extend([(ki, ki) for ki in kband_uniq])
        kptij_lst = np.asarray(kptij_lst)
    return kptij_lst
def is_j_only(kptij_lst):
    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]
    aosym_ks2 = abs(kpti-kptj).sum(axis=1) < KPT_DIFF_TOL
    j_only = np.all(aosym_ks2)
    return j_only
def loop_uniq_q(mydf, kptij_lst=None, verbose=None):
    r''' Loop over uniq q = kptj-kpti, yielding q, adapted_kptjs, adapted_ji_idx
    '''
    log = logger.new_logger(mydf, verbose)

    if kptij_lst is None:
        # kpts = mydf.kpts
        # kptij_lst = [(ki, kpts[j]) for i, ki in enumerate(kpts) for j in range(i+1)]
        # kptij_lst = np.asarray(kptij_lst)
        kptij_lst = get_kptij_lst(mydf.kpts)
    else:
        kptij_lst = np.asarray(kptij_lst).reshape(-1,2,3)
    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    uniq_kpts, uniq_index, uniq_inverse = unique(kptjs-kptis)

    ared = mydf.cell.lattice_vectors() / (2*np.pi)
    def kconserve_indices(kpt):
        '''search which (kpts+kpt) satisfies momentum conservation'''
        kdif = np.einsum('wx,ix->wi', ared, uniq_kpts + kpt)
        kdif_int = np.rint(kdif)
        mask = np.einsum('wi->i', abs(kdif - kdif_int)) < KPT_DIFF_TOL
        uniq_kptji_ids = np.where(mask)[0]
        return uniq_kptji_ids

    done = np.zeros(len(uniq_kpts), dtype=bool)
    for k, kpt in enumerate(uniq_kpts):
        if done[k]:
            continue

        uniq_kptji_ids = kconserve_indices(-kpt)
        log.debug1("Symmetry pattern (k - %s)*a= 2n pi", kpt)
        log.debug1("    make_kpt for uniq_kptji_ids %s", uniq_kptji_ids)
        for uniq_kptji_id in uniq_kptji_ids:
            if not done[uniq_kptji_id]:
                q = uniq_kpts[uniq_kptji_id]
                adapted_ji_idx = np.where(uniq_inverse == uniq_kptji_id)[0]
                adapted_kptjs = kptjs[adapted_ji_idx]
                log.debug1('adapted_ji_idx = %s', adapted_ji_idx)
                yield q, adapted_kptjs, adapted_ji_idx
        done[uniq_kptji_ids] = True

        uniq_kptji_ids = kconserve_indices(kpt)
        log.debug1("Symmetry pattern (k + %s)*a= 2n pi", kpt)
        log.debug1("    make_kpt for uniq_kptji_ids %s", uniq_kptji_ids)
        for uniq_kptji_id in uniq_kptji_ids:
            if not done[uniq_kptji_id]:
                q = uniq_kpts[uniq_kptji_id]
                adapted_ji_idx = np.where(uniq_inverse == uniq_kptji_id)[0]
                adapted_kptjs = kptjs[adapted_ji_idx]
                log.debug1('adapted_ji_idx = %s', adapted_ji_idx)
                yield q, adapted_kptjs, adapted_ji_idx
        done[uniq_kptji_ids] = True
