""" Generating MO integrals
"""
import h5py
import tempfile
import numpy as np

from pyscf import lib
from pyscf.pbc import tools

from pyscf.pbc.pwscf.pw_helper_cpwpw import get_kcomp, set_kcomp

dot = lib.dot
einsum = np.einsum


def get_molint(mf_or_fchk, kpts, nvir=None, erifile=None, dataname="eri_mo"):
    """
    Args:
        mf_or_fchk : PWSCF object or str of
    """
    pass


def kconserv(kpt123, reduce_latvec, kdota):
    tmp = dot(kpt123.reshape(1,-1), reduce_latvec) + kdota
    return np.where(abs(tmp - np.rint(tmp)).sum(axis=1)<1e-6)[0][0]


def get_molint_from_C(cell, C_ks, mo_slices, kpts, exxdiv=None,
                      erifile=None, dataname="eris"):
    """
    Args:
        C_ks : list or h5py group
            If list, the MO coeff for the k-th kpt is C_ks[k]
            If h5py, the MO coeff for the k-th kpt is C_ks["%d"%k][()]
            Note: this function assumes that MOs from different kpts are appropriately padded.
        mo_slices
        erifile: str, h5py File or h5py Group
            The file to store the ERIs. If not given, the ERIs are held in memory.
    """

    nkpts = len(kpts)
    mesh = cell.mesh
    coords = cell.get_uniform_grids(mesh=mesh)
    ngrids = coords.shape[0]
    fac = ngrids**3. / cell.vol / nkpts

    reduce_latvec = cell.lattice_vectors() / (2*np.pi)
    kdota = dot(kpts, reduce_latvec)

    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    fswap = lib.H5TmpFile(swapfile.name)
    swapfile = None

    C_ks_R = fswap.create_group("C_ks_R")
    for k in range(nkpts):
        key = "%d"%k
        C_ks_R[key] = tools.ifft(C_ks[key][()], mesh)

    dtype = np.complex128
    dsize = 16
    nmos = [mo_slice[1]-mo_slice[0] for mo_slice in mo_slices]
    buf = np.empty(nmos[0]*nmos[1]*ngrids, dtype=dtype)

    if erifile is None:
        incore = True
        deri = np.zeros((nkpts,nkpts,nkpts,*nmos), dtype=dtype)
    elif isinstance(erifile, (str, h5py.Group)):
        incore = False
        if isinstance(erifile, str):
            if h5py.is_hdf5(erifile):
                feri = h5py.File(erifile, "a")
            else:
                feri = h5py.File(erifile, "w")
        else:
            assert(isinstance(erifile, h5py.Group))
            feri = erifile
        if dataname in feri: del feri[dataname]
        deri = feri.create_dataset(dataname, (nkpts,nkpts,nkpts,*nmos),
                                   dtype=dtype)
        buf2 = np.empty(nmos, dtype=dtype)
    else:
        raise RuntimeError

    for k1 in range(nkpts):
        kpt1 = kpts[k1]
        p0,p1 = mo_slices[0]
        C_k1_R = get_kcomp(C_ks_R, k1, mo_slice=mo_slices[0])
        for k2 in range(nkpts):
            kpt2 = kpts[k2]
            kpt12 = kpt2 - kpt1
            q0,q1 = mo_slices[1]
            C_k2_R = get_kcomp(C_ks_R, k2, mo_slice=mo_slices[1])
            coulG_k12 = tools.get_coulG(cell, kpt12, exx=exxdiv, mesh=mesh)
# FIXME: batch appropriately
            v_pq_k12 = np.ndarray((nmos[0],nmos[1],ngrids), dtype=dtype,
                                  buffer=buf)
            for p in range(p0,p1):
                ip = p - p0
                v_pq_k12[ip] = tools.ifft(tools.fft(C_k1_R[ip].conj() * C_k2_R,
                                          mesh) * coulG_k12, mesh)

            for k3 in range(nkpts):
                kpt3 = kpts[k3]
                r0,r1 = mo_slices[2]
                C_k3_R = get_kcomp(C_ks_R, k3, mo_slice=mo_slices[2])
                kpt123 = kpt12 - kpt3
                k4 = kconserv(kpt123, reduce_latvec, kdota)
                kpt4 = kpts[k4]
                s0,s1 = mo_slices[3]
                C_k4_R = get_kcomp(C_ks_R, k4, mo_slice=mo_slices[3])

                kpt1234 = kpt123 + kpt4
                phase = np.exp(1j*lib.dot(coords,
                               kpt1234.reshape(-1,1))).reshape(-1)

                if incore:
                    vpqrs = deri[k1,k2,k3]
                else:
                    vpqrs = np.ndarray(nmos, dtype=dtype, buffer=buf2)
                for r in range(r0,r1):
                    ir = r - r0
                    rho_rs_k34 = C_k3_R[ir].conj() * C_k4_R * phase
                    vpqrs[:,:,ir] = dot(v_pq_k12.reshape(-1,ngrids),
                                        rho_rs_k34.T).reshape(nmos[0],nmos[1],nmos[-1])
                vpqrs *= fac
                if not incore:
                    deri[k1,k2,k3,:] = vpqrs
                    vpqrs = None

    fswap.close()

    return deri


if __name__ == "__main__":
    from pyscf.pbc import pwscf, gto

    atom = "H 0 0 0; H 0.9 0 0"
    a = np.eye(3) * 3
    basis = "gth-szv"
    pseudo = "gth-pade"

    ke_cutoff = 30

    cell = gto.Cell(atom=atom, a=a, basis=basis, pseudo=pseudo,
                    ke_cutoff=ke_cutoff)
    cell.build()
    cell.verbose = 5

    kmesh = [2,1,1]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

    nvir = 5
    chkfile = "mf.chk"
    mf = pwscf.KRHF(cell, kpts)
    mf.nvir = nvir
    # mf.init_guess = "chk"
    mf.chkfile = chkfile
    mf.kernel()

    mmp = pwscf.KMP2(mf)
    mmp.kernel()

    fchk = h5py.File(chkfile, "r")
    C_ks = fchk["mo_coeff"]

    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    erifile = swapfile.name
    swapfile = None

    no = cell.nelectron // 2
    nmo = no + nvir
    mo_slices = [(0,no),(no,nmo),(0,no),(no,nmo)]
    feri = get_molint_from_C(cell, C_ks, mo_slices, kpts, exxdiv=None,
                             erifile=erifile, dataname="eris")

    fchk.close()