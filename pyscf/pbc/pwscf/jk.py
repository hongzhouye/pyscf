""" J/K builder for PW-SCF
"""

import tempfile
import numpy as np

from pyscf.pbc import tools
from pyscf.pbc.pwscf.pw_helper import (get_kcomp, set_kcomp, acc_kcomp,
                                       scale_kcomp)
from pyscf.pbc.lib.kpts_helper import member, is_zero
from pyscf import lib
from pyscf import __config__


THR_OCC = 1e-10


def get_rho_R(C_ks, mocc_ks, mesh):
    nkpts = len(C_ks)
    rho_R = 0.
    for k in range(nkpts):
        occ = np.where(mocc_ks[k] > THR_OCC)[0].tolist()
        Co_k = get_kcomp(C_ks, k, occ=occ)
        Co_k_R = tools.ifft(Co_k, mesh)
        rho_R += np.einsum("ig,ig->g", Co_k_R.conj(), Co_k_R).real
    return rho_R


def apply_j_kpt(C_k, mesh, vj_R, C_k_R=None):
    if C_k_R is None: C_k_R = tools.ifft(C_k, mesh)
    return tools.fft(C_k_R * vj_R, mesh)


def apply_j(C_ks, mesh, vj_R, C_ks_R=None, out=None):
    nkpts = len(C_ks)
    if out is None: out = [None] * nkpts
    for k in range(nkpts):
        C_k = get_kcomp(C_ks, k)
        C_k_R is None if C_ks_R is None else get_kcomp(C_ks_R, k)
        Cbar_k = apply_j_kpt(C_k, mesh, vj_R, C_k_R=C_k_R)
        set_kcomp(Cbar_k, out, k)

    return out


def apply_k_kpt(cell, C_k, kpt1, C_ks, mocc_ks, kpts, mesh, Gv,
                C_k_R=None, C_ks_R=None, exxdiv=None):
    r""" Apply the EXX operator to given MOs

    Math:
        Cbar_k(G) = \sum_{j,k'} \sum_{G'} rho_{jk',ik}(G') v(k-k'+G') C_k(G-G')
    Code:
        rho_r = C_ik_r * C_jk'_r.conj()
        rho_G = FFT(rho_r)
        coulG = get_coulG(k-k')
        v_r = iFFT(rho_G * coulG)
        Cbar_ik_G = FFT(v_r * C_jk'_r)
    """
    ngrids = Gv.shape[0]
    nkpts = len(kpts)
    fac = ngrids**2./(cell.vol*nkpts)

    Cbar_k = np.zeros_like(C_k)
    if C_k_R is None: C_k_R = tools.ifft(C_k, mesh)

    for k2 in range(nkpts):
        kpt2 = kpts[k2]
        coulG = tools.get_coulG(cell, kpt1-kpt2, exx=False, mesh=mesh, Gv=Gv)

        occ = np.where(mocc_ks[k2]>THR_OCC)[0]
        no_k2 = occ.size
        if C_ks_R is None:
            Co_k2 = get_kcomp(C_ks, k2, occ=occ)
            Co_k2_R = tools.ifft(Co_k2, mesh)
            Co_k2 = None
        else:
            Co_k2_R = get_kcomp(C_ks_R, k2, occ=occ)
        for j in range(no_k2):
            Cj_k2_R = Co_k2_R[j]
            vij_R = tools.ifft(
                tools.fft(C_k_R * Cj_k2_R.conj(), mesh) * coulG, mesh)
            Cbar_k += vij_R * Cj_k2_R

    Cbar_k = tools.fft(Cbar_k, mesh) * fac

    return Cbar_k


def apply_k_kpt_support_vec(C_k, W_k):
    Cbar_k = lib.dot(lib.dot(C_k, W_k.conj().T), W_k)
    return Cbar_k


def apply_k_s1(cell, C_ks, Ct_ks, mocc_ks, kpts, mesh, Gv, out=None):
    nkpts = len(kpts)
    ngrids = np.prod(mesh)
    fac = ngrids**2./(cell.vol*nkpts)
    occ_ks = [np.where(mocc_ks[k] > THR_OCC)[0] for k in range(nkpts)]

    if out is None: out = [None] * nkpts

# swap file to hold FFTs
    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    fswap = lib.H5TmpFile(swapfile.name)
    swapfile = None

    Co_ks_R = fswap.create_group("Co_ks_R")
    Ct_ks_R = fswap.create_group("Ct_ks_R")

    for k in range(nkpts):
        Co_k = get_kcomp(C_ks, k, occ=occ_ks[k])
        set_kcomp(tools.ifft(Co_k, mesh), Co_ks_R, k)
        Co_k = None

    for k in range(nkpts):
        Ct_k = get_kcomp(Ct_ks, k)
        set_kcomp(tools.ifft(Ct_k, mesh), Ct_ks_R, k)
        Ct_k = None

    for k1,kpt1 in enumerate(kpts):
        Ct_k1_R = get_kcomp(Ct_ks_R, k1)
        Ctbar_k1 = np.zeros_like(Ct_k1_R)
        for k2,kpt2 in enumerate(kpts):
            coulG = tools.get_coulG(cell, kpt1-kpt2, exx=False, mesh=mesh,
                                    Gv=Gv)
            Co_k2_R = get_kcomp(Co_ks_R, k2)
            for j in range(no_ks[k2]):
                Cj_k2_R = Co_k2_R[j]
                vij_R = tools.ifft(tools.fft(Ct_k1_R * Cj_k2_R.conj(), mesh) *
                                   coulG, mesh)
                Ctbar_k1 += vij_R * Cj_k2_R

        Ctbar_k1 = tools.fft(Ctbar_k1, mesh) * fac
        set_kcomp(out, Ctbar_k1, k)
        Ctbar_k1 = None

    return out


def apply_k_s2(cell, C_ks, mocc_ks, kpts, mesh, Gv, out):
    nkpts = len(kpts)
    ngrids = np.prod(mesh)
    fac = ngrids**2./(cell.vol*nkpts)
    occ_ks = [np.where(mocc_ks[k] > THR_OCC)[0] for k in range(nkpts)]

    if out is None: out = [None] * nkpts

    if isinstance(C_ks, list):
        n_ks = [C_ks[k].shape[0] for k in range(nkpts)]
    else:
        n_ks = [C_ks["%d"%k].shape[0] for k in range(nkpts)]
    no_ks = [np.sum(mocc_ks[k]>THR_OCC) for k in range(nkpts)]
    n_max = np.max(n_ks)
    no_max = np.max(no_ks)

# TODO: non-aufbau configurations
    for k in range(nkpts):
        if np.sum(mocc_ks[k][:no_ks[k]]>THR_OCC) != no_ks[k]:
            raise NotImplementedError("Non-aufbau configurations are not supported.")

    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    fswap = lib.H5TmpFile(swapfile.name)
    swapfile = None

    C_ks_R = fswap.create_group("C_ks_R")

    for k in range(nkpts):
        C_k = get_kcomp(C_ks, k)
        set_kcomp(tools.ifft(C_k, mesh), C_ks_R, k)
        set_kcomp(np.zeros_like(C_k), out, k)
        C_k = None

    dtype = np.complex128

    buf1 = np.empty(n_max*ngrids, dtype=dtype)
    buf2 = np.empty(no_max*ngrids, dtype=dtype)
    for k1,kpt1 in enumerate(kpts):
        C_k1_R = get_kcomp(C_ks_R, k1)
        no_k1 = no_ks[k1]
        n_k1 = n_ks[k1]
        Cbar_k1 = np.ndarray((n_k1,ngrids), dtype=dtype, buffer=buf1)
        Cbar_k1.fill(0)
        for k2,kpt2 in enumerate(kpts):
            if n_k1 == no_k1 and k2 > k1: continue

            C_k2_R = get_kcomp(C_ks_R, k2)
            no_k2 = no_ks[k2]

            coulG = tools.get_coulG(cell, kpt1-kpt2, exx=False, mesh=mesh,
                                    Gv=Gv)

            # o --> o
            if k2 <= k1:
                Cbar_k2 = np.ndarray((no_k2,ngrids), dtype=dtype, buffer=buf2)
                Cbar_k2.fill(0)

                for i in range(no_k1):
                    jmax = i+1 if k2 == k1 else no_k2
                    jmax2 = jmax-1 if k2 == k1 else jmax
                    vji_R = tools.ifft(tools.fft(C_k2_R[:jmax].conj() *
                                       C_k1_R[i], mesh) * coulG, mesh)
                    Cbar_k1[i] += np.sum(vji_R * C_k2_R[:jmax], axis=0)
                    Cbar_k2[:jmax2] += vji_R[:jmax2].conj() * C_k1_R[i]

                acc_kcomp(Cbar_k2, out, k2, occ=occ_ks[k2])

            # o --> v
            if n_k1 > no_k1:
                for j in range(no_ks[k2]):
                    vij_R = tools.ifft(tools.fft(C_k1_R[no_k1:] *
                                                 C_k2_R[j].conj(), mesh) *
                                       coulG, mesh)
                    Cbar_k1[no_k1:] += vij_R  * C_k2_R[j]

        acc_kcomp(Cbar_k1, out, k1)

    for k in range(nkpts):
        set_kcomp(tools.fft(get_kcomp(out, k), mesh) * fac, out, k)

    return out


def apply_k(cell, C_ks, mocc_ks, kpts, mesh, Gv, Ct_ks=None, exxdiv=None,
            out=None):
    if Ct_ks is None:
        return apply_k_s2(cell, C_ks, mocc_ks, kpts, mesh, Gv, out)
    else:
        return apply_k_s1(cell, C_ks, Ct_ks, mocc_ks, kpts, mesh, Gv, out)


def jk(mf, with_jk=None, ace_exx=True):
    if with_jk is None:
        with_jk = PWJK(mf.cell, mf.kpts, exxdiv=mf.exxdiv)
        with_jk.ace_exx = ace_exx

    mf.with_jk = with_jk

    return mf


class PWJK:

    ace_exx = getattr(__config__, "pbc_pwscf_jk_PWJK_ace_exx", True)

    def __init__(self, cell, kpts, mesh=None, exxdiv=None):
        self.cell = cell
        self.kpts = kpts
        if mesh is None: mesh = cell.mesh
        self.mesh = mesh
        self.Gv = cell.get_Gv(mesh)
        self.exxdiv = exxdiv

        # the following are not input options
        self.swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        self.fswap = lib.H5TmpFile(self.swapfile.name)
        self.exx_W_ks = self.fswap.create_group("exx_W_ks")

    def get_Gv(self, mesh):
        if is_zero(np.asarray(mesh)-np.asarray(self.mesh)):
            return self.Gv
        else:
            return self.cell.get_Gv(mesh)

    def get_vj_R(self, C_ks, mocc_ks, mesh=None, Gv=None, ncomp=1):
        if mesh is None: mesh = self.mesh
        if Gv is None: Gv = self.get_Gv(mesh)
        if ncomp == 1:
            rho_R = get_rho_R(C_ks, mocc_ks, mesh)
        else:
            rho_R = 0.
            for comp in range(ncomp):
                C_ks_comp = get_kcomp(C_ks, comp, load=False)
                rho_R += get_rho_R(C_ks_comp, mocc_ks[comp], mesh)
            rho_R *= 1./ncomp

        cell = self.cell
        nkpts = len(self.kpts)
        ngrids = Gv.shape[0]
        fac = ngrids**2 / (cell.vol*nkpts)
        vj_R = tools.ifft(tools.fft(rho_R, mesh) * tools.get_coulG(cell, Gv=Gv),
                          mesh).real * fac

        return vj_R

    def update_k_support_vec(self, C_ks, mocc_ks, kpts, Ct_ks=None,
                             mesh=None, Gv=None, exxdiv=None, comp=None):
        nkpts = len(kpts)

        if comp is None:
            out = self.exx_W_ks
        elif isinstance(comp, int):
            keycomp = "%d" % comp
            if not keycomp in self.exx_W_ks:
                self.exx_W_ks.create_group(keycomp)
            out = self.exx_W_ks[keycomp]
        else:
            raise RuntimeError("comp must be None or int")

        if self.ace_exx:
            from pyscf.pbc.pwscf.pseudo import get_support_vec
            if mesh is None: mesh = self.mesh
            if Gv is None: Gv = self.get_Gv(mesh)

            dname0 = "W_ks"
            if dname0 in self.fswap: del self.fswap[dname0]
            W_ks = self.fswap.create_group(dname0)
            self.apply_k(C_ks, mocc_ks, kpts, Ct_ks=Ct_ks,
                         mesh=mesh, Gv=Gv, exxdiv=exxdiv, out=W_ks)

            for k in range(nkpts):
                if Ct_ks is None:
                    Ct_k = get_kcomp(C_ks, k)
                else:
                    Ct_k = get_kcomp(Ct_ks, k)
                W_k = get_kcomp(W_ks, k)
                W_k = get_support_vec(Ct_k, W_k, method="cd")
                set_kcomp(W_k, out, k)
                W_k = None

            del self.fswap[dname0]
        else:   # store ifft of Co_ks
            if mesh is None: mesh = self.mesh
            for k in range(nkpts):
                occ = np.where(mocc_ks[k]>THR_OCC)[0]
                Co_k = get_kcomp(C_ks, k, occ=occ)
                set_kcomp(tools.ifft(Co_k, mesh), out, k)

    def apply_j_kpt(self, C_k, mesh=None, vj_R=None, C_k_R=None):
        if mesh is None: mesh = self.mesh
        if vj_R is None: vj_R = self.vj_R
        return apply_j_kpt(C_k, mesh, vj_R, C_k_R=None)

    def apply_j(self, C_ks, mesh=None, vj_R=None, C_ks_R=None, out=None):
        if mesh is None: mesh = self.mesh
        if vj_R is None: vj_R = self.vj_R
        return apply_j(C_ks, mesh, vj_R, C_ks_R=out, out=out)

    def apply_k_kpt(self, C_k, kpt, mesh=None, Gv=None, exxdiv=None, comp=None):
        if comp is None:
            W_ks = self.exx_W_ks
        elif isinstance(comp, int):
            W_ks = get_kcomp(self.exx_W_ks, comp, load=False)
        else:
            raise RuntimeError("comp must be None or int.")

        if self.ace_exx:
            k = member(kpt, self.kpts)[0]
            W_k = get_kcomp(W_ks, k)
            return apply_k_kpt_support_vec(C_k, W_k)
        else:
            cell = self.cell
            kpts = self.kpts
            nkpts = len(kpts)
            if mesh is None: mesh = self.mesh
            if Gv is None: Gv = self.get_Gv(mesh)
            if exxdiv is None: exxdiv = self.exxdiv
            mocc_ks = [np.ones(get_kcomp(W_ks, k, load=False).shape[0])*2
                       for k in range(nkpts)]
            return apply_k_kpt(cell, C_k, kpt, None, mocc_ks, kpts, mesh, Gv,
                               C_ks_R=W_ks, exxdiv=exxdiv)

    def apply_k(self, C_ks, mocc_ks, kpts, Ct_ks=None, mesh=None, Gv=None,
                exxdiv=None, out=None):
        cell = self.cell
        if mesh is None: mesh = self.mesh
        if Gv is None: Gv = self.get_Gv(mesh)
        if exxdiv is None: exxdiv = self.exxdiv
        return apply_k(cell, C_ks, mocc_ks, kpts, mesh, Gv, Ct_ks=Ct_ks,
                       exxdiv=exxdiv, out=out)