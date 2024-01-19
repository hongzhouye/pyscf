#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

"""
DIIS
"""

from functools import reduce
import numpy
import scipy.linalg
import scipy.optimize
from pyscf import lib
from pyscf.lib import logger

DEBUG = False

# J. Mol. Struct. 114, 31-34 (1984); DOI:10.1016/S0022-2860(84)87198-7
# PCCP, 4, 11 (2002); DOI:10.1039/B108658H
# GEDIIS, JCTC, 2, 835 (2006); DOI:10.1021/ct050275a
# C2DIIS, IJQC, 45, 31 (1993); DOI:10.1002/qua.560450106
# SCF-EDIIS, JCP 116, 8255 (2002); DOI:10.1063/1.1470195

# error vector = SDF-FDS
# error vector = F_ai ~ (S-SDS)*S^{-1}FDS = FDS - SDFDS ~ FDS-SDF in converge
class CDIIS(lib.diis.DIIS):
    def __init__(self, mf=None, filename=None, Corth=None):
        lib.diis.DIIS.__init__(self, mf, filename)
        self.rollback = 0
        self.space = 8
        self.Corth = Corth
        #?self._scf = mf
        #?if hasattr(self._scf, 'get_orbsym'): # Symmetry adapted SCF objects
        #?    self.orbsym = mf.get_orbsym(Corth)
        #?    sym_forbid = self.orbsym[:,None] != self.orbsym

    def update(self, s, d, f, *args, **kwargs):
        errvec = get_err_vec(s, d, f, self.Corth)
        logger.debug1(self, 'diis-norm(errvec)=%g', numpy.linalg.norm(errvec))
        xnew = lib.diis.DIIS.update(self, f, xerr=errvec)
        if self.rollback > 0 and len(self._bookkeep) == self.space:
            self._bookkeep = self._bookkeep[-self.rollback:]
        return xnew

    def get_num_vec(self):
        if self.rollback:
            return self._head
        else:
            return len(self._bookkeep)

SCFDIIS = SCF_DIIS = DIIS = CDIIS

def get_err_vec_orig(s, d, f):
    '''error vector = SDF - FDS'''
    if isinstance(f, numpy.ndarray) and f.ndim == 2:
        sdf = reduce(numpy.dot, (s,d,f))
        errvec = (sdf.conj().T - sdf).ravel()

    elif isinstance(f, numpy.ndarray) and f.ndim == 3 and s.ndim == 3:
        errvec = []
        for i in range(f.shape[0]):
            sdf = reduce(numpy.dot, (s[i], d[i], f[i]))
            errvec.append((sdf.conj().T - sdf).ravel())
        errvec = numpy.hstack(errvec)

    elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
        errvec = numpy.hstack([
            get_err_vec_orig(s, d[0], f[0]).ravel(),
            get_err_vec_orig(s, d[1], f[1]).ravel()])
    else:
        raise RuntimeError('Unknown SCF DIIS type')
    return errvec

def get_err_vec_orth(s, d, f, Corth):
    '''error vector in orthonormal basis = C.T.conj() (SDF - FDS) C'''
    # Symmetry information to reduce numerical error in DIIS (issue #1524)
    orbsym = getattr(Corth, 'orbsym', None)
    if orbsym is not None:
        sym_forbid = orbsym[:,None] != orbsym

    if isinstance(f, numpy.ndarray) and f.ndim == 2:
        sdf = reduce(numpy.dot, (Corth.conj().T, s, d, f, Corth))
        if orbsym is not None:
            sdf[sym_forbid] = 0
        errvec = (sdf.conj().T - sdf).ravel()

    elif isinstance(f, numpy.ndarray) and f.ndim == 3 and s.ndim == 3:
        errvec = []
        for i in range(f.shape[0]):
            sdf = reduce(numpy.dot, (Corth[i].conj().T, s[i], d[i], f[i], Corth[i]))
            if orbsym is not None:
                sdf[sym_forbid] = 0
            errvec.append((sdf.conj().T - sdf).ravel())
        errvec = numpy.hstack(errvec)

    elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
        errvec = numpy.hstack([
            get_err_vec_orth(s, d[0], f[0], Corth[0]).ravel(),
            get_err_vec_orth(s, d[1], f[1], Corth[1]).ravel()])
    else:
        raise RuntimeError('Unknown SCF DIIS type')
    return errvec

def get_err_vec(s, d, f, Corth=None):
    if Corth is None:
        return get_err_vec_orig(s, d, f)
    else:
        return get_err_vec_orth(s, d, f, Corth)

class CDIIS1(lib.diis.DIIS):
    '''error vector = f(n) - f(n-1)'''
    def __init__(self, mf=None, filename=None, Corth=None):
        lib.diis.DIIS.__init__(self, mf, filename)
        self.space = 8
        self.Corth = Corth
    def update(self, s, d, f, *args, **kwargs):
        if self.Corth is not None:
            c = self.Corth
            f_ = reduce(numpy.dot, (c.T.conj(), f, c))
        else:
            f_ = f
        xnew = lib.diis.DIIS.update(self, f_)
        if self._H is not None:
            errvec = self.get_err_vec(self._head-1)
            logger.debug1(self, 'diis-norm(errvec)=%g', numpy.linalg.norm(errvec))
        if self.Corth is not None:
            c = self.Corth
            xnew = reduce(numpy.dot, (s, c, xnew, c.T.conj(), s))
        return xnew

#####
class DIISBase(lib.StreamObject):

    update_type = 1
    lindep_thresh = 1e-10
    err_remove_damp = 1e-10

    def __init__(self, dev=None):
        if dev is not None:
            self.verbose = dev.verbose
            self.stdout = dev.stdout
        else:
            self.verbose = logger.INFO
            self.stdout = sys.stdout

        self.space = 8
        self.damp = 0

        # internal variables
        self._xs = None
        self._fs = None
        self._es = None

    def _update(self, x, f, e=None):
        if self._xs is None:
            self._xs = []
            self._fs = []
            self._es = []

        if e is None:
            e = (f - x).ravel()
        logger.debug1(self, 'diis-norm(errvec)=%g', numpy.linalg.norm(e))

        damp = self.damp
        if abs(self.damp) > 1e-10:
            if numpy.linalg.norm(e) < self.err_remove_damp:
                logger.debug1(self, 'removing damping from diis')
                damp = 0

        n = len(self._xs)
        if n < self.space:
            self._xs.append( x )
            self._fs.append( f )
            self._es.append( e )
        else:
            for i in range(n-1):
                self._xs[i] = self._xs[i+1]
                self._fs[i] = self._fs[i+1]
                self._es[i] = self._es[i+1]
            self._xs[-1] = x
            self._fs[-1] = f
            self._es[-1] = e

        xnew = self.extrapolate(damp)
        if xnew is not None:
            return xnew

        # restart
        self.reset()
        return self._update(x, f, e)

    def update(self, x, f, e=None):
        return self._update(x, f, e)

    def extrapolate(self, damp=None):
        if damp is None: damp = self.damp

        n = len(self._xs)
        if n == 1:
            if abs(damp) < 1e-10:
                return self._fs[0]
            else:
                return self._fs[0] - damp * (self._fs[0] - self._xs[0])

        dtype = self._fs[0].dtype
        A = numpy.zeros((n,n), dtype=dtype)
        for i in range(n):
            for j in range(i+1):
                a = numpy.dot(self._es[i], self._es[j])
                A[i,j] = A[j,i] = a
        B = numpy.zeros((n+1,n+1), dtype=dtype)
        B[:n,:n] =  A
        B[:n, n] = -1
        B[ n,:n] = -1
        b = numpy.zeros(n+1, dtype=dtype)
        b[n] = -1

        w, v = scipy.linalg.eigh(B)
        logger.debug1(self, 'diis-eigval %s', numpy.sort(abs(w)))
        if numpy.any(abs(w) < self.lindep_thresh):
            logger.debug(self, 'Linear dependence found in DIIS error vectors.')
            idx = abs(w) > self.lindep_thresh
            logger.debug(self, 'Keeping %d/%d diis vectors', numpy.count_nonzero(idx), idx.size)
            alps = numpy.dot(v[:,idx]*(1./w[idx]), numpy.dot(v[:,idx].T.conj(), b))[:n]
        else:
            try:
                alps = numpy.linalg.solve(B, b)[:n]
            except numpy.linalg.linalg.LinAlgError as e:
                logger.warn(self, ' diis singular, eigh(h) %s', w)
                raise e
        logger.debug1(self, 'diis-c %s', alps)

        if abs(damp) < 1e-10:
            if self.update_type == 1:
                xnew = sum([alps[i]*self._xs[i] for i in range(n)])
            elif self.update_type == 2:
                xnew = sum([alps[i]*self._fs[i] for i in range(n)])
            else:
                raise RuntimeError
        else:
            xnew = sum([alps[i]*(self._fs[i] - damp*(self._fs[i]-self._xs[i]))
                        for i in range(n)])

        return xnew

    def reset(self):
        self._xs = None
        self._fs = None
        self._es = None

class DIIS_Fock_Res(DIISBase):

    def __init__(self, mf, Corth):
        DIISBase.__init__(self, mf)
        self.Corth = Corth

    def update(self, s, d, f, f_last):
        ''' f_last --[eigh]--> d --[F-build]--> f
        '''
        c = self.Corth
        if isinstance(f, numpy.ndarray) and f.ndim == 2:
            f1 = reduce(numpy.dot, (c.T.conj(), f, c))
            f1_last = reduce(numpy.dot, (c.T.conj(), f_last, c))
        elif isinstance(f, numpy.ndarray) and f.ndim == 3:
            f1 = lib.asarray([reduce(numpy.dot, (ck.T.conj(), fk, ck)) for ck,fk in zip(c,f)])
            f1_last = lib.asarray([reduce(numpy.dot, (ck.T.conj(), fk, ck))
                                   for ck,fk in zip(c,f_last)])
        else:
            raise RuntimeError('Unknown SCF DIIS type')
        f1new = self._update(f1_last, f1)
        if isinstance(f, numpy.ndarray) and f.ndim == 2:
            sc = numpy.dot(s, c)
            fnew = reduce(numpy.dot, (sc, f1new, sc.T.conj()))
        elif isinstance(f, numpy.ndarray) and f.ndim == 3 and s.ndim == 3:
            fnew = []
            for sk,ck,f1k in zip(s,c,f1new):
                sc = numpy.dot(sk, ck)
                fnew.append( reduce(numpy.dot, (sc, f1k, sc.T.conj())) )
            fnew = lib.asarray(fnew)
        elif isinstance(f, numpy.ndarray) and f.ndim == 3 and s.ndim == 2:
            fnew = []
            for ck,f1k in zip(c,f1new):
                sc = numpy.dot(s, ck)
                fnew.append( reduce(numpy.dot, (sc, f1k, sc.T.conj())) )
            fnew = lib.asarray(fnew)
        else:
            raise RuntimeError('Unknown SCF DIIS type')
        return fnew

class DIIS_Fock_Res_TypeI(DIIS_Fock_Res):
    update_type = 1
class DIIS_Fock_Res_TypeII(DIIS_Fock_Res):
    update_type = 2


class DIIS_Fock_Err(DIISBase):

    def __init__(self, mf, Corth):
        DIISBase.__init__(self, mf)
        self.Corth = Corth

    def update(self, s, d, f, f_last):
        ''' f_last --[eigh]--> d --[F-build]--> f
        '''
        from pyscf.scf.diis import get_err_vec
        e = get_err_vec(s, d, f, self.Corth)
        fnew = self._update(f_last, f, e)
        return fnew

class DIIS_Fock_Err_TypeI(DIIS_Fock_Err):
    update_type = 1
class DIIS_Fock_Err_TypeII(DIIS_Fock_Err):
    update_type = 2
#####

class EDIIS(lib.diis.DIIS):
    '''SCF-EDIIS
    Ref: JCP 116, 8255 (2002); DOI:10.1063/1.1470195
    '''
    def update(self, s, d, f, mf, h1e, vhf):
        if self._head >= self.space:
            self._head = 0
        if not self._buffer:
            shape = (self.space,) + f.shape
            self._buffer['dm'  ] = numpy.zeros(shape, dtype=f.dtype)
            self._buffer['fock'] = numpy.zeros(shape, dtype=f.dtype)
            self._buffer['etot'] = numpy.zeros(self.space)
        self._buffer['dm'  ][self._head] = d
        self._buffer['fock'][self._head] = f
        self._buffer['etot'][self._head] = mf.energy_elec(d, h1e, vhf)[0]
        self._head += 1

        ds = self._buffer['dm'  ]
        fs = self._buffer['fock']
        es = self._buffer['etot']
        etot, c = ediis_minimize(es, ds, fs)
        logger.debug1(self, 'E %s  diis-c %s', etot, c)
        fock = numpy.einsum('i,i...pq->...pq', c, fs)
        return fock

def ediis_minimize(es, ds, fs):
    nx = es.size
    nao = ds.shape[-1]
    ds = ds.reshape(nx,-1,nao,nao)
    fs = fs.reshape(nx,-1,nao,nao)
    df = numpy.einsum('inpq,jnqp->ij', ds, fs).real
    diag = df.diagonal()
    df = diag[:,None] + diag - df - df.T

    def costf(x):
        c = x**2 / (x**2).sum()
        return numpy.einsum('i,i', c, es) - numpy.einsum('i,ij,j', c, df, c)

    def grad(x):
        x2sum = (x**2).sum()
        c = x**2 / x2sum
        fc = es - 2*numpy.einsum('i,ik->k', c, df)
        cx = numpy.diag(x*x2sum) - numpy.einsum('k,n->kn', x**2, x)
        cx *= 2/x2sum**2
        return numpy.einsum('k,kn->n', fc, cx)

    if DEBUG:
        x0 = numpy.random.random(nx)
        dfx0 = numpy.zeros_like(x0)
        for i in range(nx):
            x1 = x0.copy()
            x1[i] += 1e-4
            dfx0[i] = (costf(x1) - costf(x0))*1e4
        print((dfx0 - grad(x0)) / dfx0)

    res = scipy.optimize.minimize(costf, numpy.ones(nx), method='BFGS',
                                  jac=grad, tol=1e-9)
    return res.fun, (res.x**2)/(res.x**2).sum()


class ADIIS(lib.diis.DIIS):
    '''
    Ref: JCP 132, 054109 (2010); DOI:10.1063/1.3304922
    '''
    def update(self, s, d, f, mf, h1e, vhf):
        if self._head >= self.space:
            self._head = 0
        if not self._buffer:
            shape = (self.space,) + f.shape
            self._buffer['dm'  ] = numpy.zeros(shape, dtype=f.dtype)
            self._buffer['fock'] = numpy.zeros(shape, dtype=f.dtype)
        self._buffer['dm'  ][self._head] = d
        self._buffer['fock'][self._head] = f

        ds = self._buffer['dm'  ]
        fs = self._buffer['fock']
        fun, c = adiis_minimize(ds, fs, self._head)
        if self.verbose >= logger.DEBUG1:
            etot = mf.energy_elec(d, h1e, vhf)[0] + fun
            logger.debug1(self, 'E %s  diis-c %s ', etot, c)
        fock = numpy.einsum('i,i...pq->...pq', c, fs)
        self._head += 1
        return fock

def adiis_minimize(ds, fs, idnewest):
    nx = ds.shape[0]
    nao = ds.shape[-1]
    ds = ds.reshape(nx,-1,nao,nao)
    fs = fs.reshape(nx,-1,nao,nao)
    df = numpy.einsum('inpq,jnqp->ij', ds, fs).real
    d_fn = df[:,idnewest]
    dn_f = df[idnewest]
    dn_fn = df[idnewest,idnewest]
    dd_fn = d_fn - dn_fn
    df = df - d_fn[:,None] - dn_f + dn_fn

    def costf(x):
        c = x**2 / (x**2).sum()
        return (numpy.einsum('i,i', c, dd_fn) * 2 +
                numpy.einsum('i,ij,j', c, df, c))

    def grad(x):
        x2sum = (x**2).sum()
        c = x**2 / x2sum
        fc = 2*dd_fn
        fc+= numpy.einsum('j,kj->k', c, df)
        fc+= numpy.einsum('i,ik->k', c, df)
        cx = numpy.diag(x*x2sum) - numpy.einsum('k,n->kn', x**2, x)
        cx *= 2/x2sum**2
        return numpy.einsum('k,kn->n', fc, cx)

    if DEBUG:
        x0 = numpy.random.random(nx)
        dfx0 = numpy.zeros_like(x0)
        for i in range(nx):
            x1 = x0.copy()
            x1[i] += 1e-4
            dfx0[i] = (costf(x1) - costf(x0))*1e4
        print((dfx0 - grad(x0)) / dfx0)

    res = scipy.optimize.minimize(costf, numpy.ones(nx), method='BFGS',
                                  jac=grad, tol=1e-9)
    return res.fun, (res.x**2)/(res.x**2).sum()
