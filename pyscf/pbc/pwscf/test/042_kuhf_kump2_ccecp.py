""" Check PW-KUHF, PW-KUMP2 and read init guess from chkfile
"""


import h5py
import tempfile
import numpy as np

from pyscf.pbc import gto, pwscf
from pyscf import lib


if __name__ == "__main__":
    nk = 1
    ke_cutoff = 30
    pseudo = "ccecp"
    atom = "C 0 0 0"
    a = np.eye(3) * 4   # atom in a cubic box

# cell
    cell = gto.Cell(
        atom=atom,
        a=a,
        spin=2, # triplet
        basis="gth-szv",
        pseudo=pseudo,
        ke_cutoff=ke_cutoff
    )
    cell.build()
    cell.verbose = 6

# kpts
    kmesh = [nk]*3
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

# tempfile
    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    chkfile = swapfile.name
    swapfile = None

# krhf
    pwmf = pwscf.KUHF(cell, kpts)
    pwmf.nvir = 4 # request 4 virtual states
    pwmf.chkfile = chkfile
    pwmf.kernel()

    assert(abs(pwmf.e_tot - -5.353419724479592) < 1.e-6)

# krhf init from chkfile
    pwmf.init_guess = "chkfile"
    pwmf.kernel()

# input C0
    with h5py.File(pwmf.chkfile, "r") as f:
        C0 = [[f["mo_coeff/%d/%s"%(s,k)][()] for k in range(nkpts)]
              for s in [0,1]]
    pwmf.kernel(C0=C0)

# krmp2
    pwmp = pwscf.KUMP2(pwmf)
    pwmp.kernel()

    assert(abs(pwmp.e_corr - -0.00668294448263389) < 1.e-6)
