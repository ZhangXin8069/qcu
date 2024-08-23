from pyquda.utils import gauge_utils
from pyquda.field import LatticeFermion
from pyquda.enum_quda import QudaParity
from pyquda import init, core, quda, pyqcu, mpi
import os
import sys
from time import perf_counter
import cupy as cp
import numpy as np
test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))
os.environ["QUDA_RESOURCE_PATH"] = ".cache"
latt_size = [32, 32, 32, 32]
# latt_size = [32, 32, 32, 64]
# latt_size = [16, 16, 16, 16]
# latt_size = [8, 16, 16, 16]
# latt_size = [8, 4, 8, 64]
# latt_size = [4, 16, 16, 32]
# latt_size = [8, 16, 16, 16]
# latt_size = [16, 32, 32, 64]
# latt_size = [4, 4, 4, 4]
# latt_size = [8, 8, 8, 8]
# latt_size = [8, 8, 8, 16]
grid_size = [2, 1, 1, 2]
Lx, Ly, Lz, Lt = latt_size
Nd, Ns, Nc = 4, 4, 3
Gx, Gy, Gz, Gt = grid_size
latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
mpi.init(grid_size)
a = 1
xi_0, nu = 1, 1
coeff_r, coeff_t = 1, 1
mass = -3.5
# kappa = 1 / (2*mass+8)
# generate a vector p randomly
p = LatticeFermion(latt_size, cp.random.randn(
    Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128))
Mp = LatticeFermion(latt_size)
Mp1 = LatticeFermion(latt_size)
U = gauge_utils.gaussGauge(latt_size, 0)
# Set parameters in Dslash and use m=-3.5 to make kappa=1
# dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)
# Generate gauge and then load it
dslash = core.getDslash(latt_size, mass, 1e-9, 1000, xi_0, nu,
                        coeff_t, coeff_r, multigrid=False, anti_periodic_t=False)
# dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)
dslash.loadGauge(U)


def compare(round):
    print('===============round ', round, '======================')
    print("######p[0,0,0,1]:\n", p.lexico()[0, 0, 0, 1])
    cp.cuda.runtime.deviceSynchronize()
    t1 = perf_counter()
    quda.dslashQuda(Mp.even_ptr, p.odd_ptr, dslash.invert_param,
                    QudaParity.QUDA_EVEN_PARITY)
    quda.dslashQuda(Mp.odd_ptr, p.even_ptr, dslash.invert_param,
                    QudaParity.QUDA_ODD_PARITY)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    print(f'Quda dslash: {t2 - t1} sec')
    # then execute my code
    param = pyqcu.QcuParam()
    param.lattice_size = latt_size
    cp.cuda.runtime.deviceSynchronize()
    t1 = perf_counter()
    pyqcu.dslashCloverQcu(Mp1.even_ptr, p.odd_ptr, U.data_ptr, param, 0)
    pyqcu.dslashCloverQcu(Mp1.odd_ptr, p.even_ptr, U.data_ptr, param, 1)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    print("######Mp[0,0,0,1]:\n", Mp.lexico()[0, 0, 0, 1])
    print("######Mp1[0,0,0,1]:\n", Mp1.lexico()[0, 0, 0, 1])
    print(f'QCU dslash: {t2 - t1} sec')
    print(f'rank {0} my x and x difference: {cp.linalg.norm(Mp1.data - Mp.data) / cp.linalg.norm(Mp.data)}, takes {t2 - t1} sec, my_norm = {cp.linalg.norm(Mp1.data)}, norm = {cp.linalg.norm(Mp.data)}')
#     print("######", Mp.lexico().shape)
#     diff_x = np.abs((Mp1.lexico()-Mp.lexico()).real)
#     diff = np.sum(diff_x, axis=(-1, -2))
#     _ = np.where(diff > 1e-5)
#     print("######", diff.shape)
#     print("######T:", _[0], ",\n", len(_[0]))
#     print("######Z:", _[1], ",\n", len(_[1]))
#     print("######Y:", _[2], ",\n", len(_[2]))
#     print("######X:", _[3], ",\n", len(_[3]))
#     print("######diff_x[0,0,0,0]:\n",
#           diff_x[0, 0, 0, 1])
#     print("######diff_x[0,0,0,1]:\n",
#           diff_x[0, 0, 0, 1])
#     print("######diff_x[0,0,1,1]:\n",
#           diff_x[0, 0, 1, 1])
#     print("######diff_x[2,2,2,2]:\n",
#           diff_x[2, 2, 2, 2])
#     print("######diff_x[-1,-1,-1,-1]:\n",
#           diff_x[-1, -1, -1, -1])
#     print("######diff_x[-2,-2,-2,-2]:\n",
#           diff_x[-2, -2, -2, -2])


for i in range(0, 5):
    compare(i)
