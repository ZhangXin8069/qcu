import numpy as np
from pyquda.utils import gauge_utils
from pyquda.field import LatticeFermion
from pyquda.enum_quda import QudaParity
from pyquda import init, core, quda, pyqcu, mpi
import os
import sys
from time import perf_counter
import cupy as cp
test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))
np.set_printoptions(threshold=np.inf)
os.environ["QUDA_RESOURCE_PATH"] = ".cache"
latt_size = [32, 32, 32, 64]
latt_size = [8, 8, 8, 16]
grid_size = [1, 1, 1, 1]
Lx, Ly, Lz, Lt = latt_size
Nd, Ns, Nc = 4, 4, 3
Gx, Gy, Gz, Gt = grid_size
latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
mpi.init(grid_size)


def compare(round):
    # generate a vector p randomly
    p = LatticeFermion(latt_size, cp.random.randn(
        Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128))
    Mp = LatticeFermion(latt_size)
    Mp1 = LatticeFermion(latt_size)
    Mp2 = LatticeFermion(latt_size)
    print('===============round ', round, '======================')
    # Set parameters in Dslash and use m=-3.5 to make kappa=1
    dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)
    # Generate gauge and then load it
    U = gauge_utils.gaussGauge(latt_size, round)
    dslash.loadGauge(U)
    cp.cuda.runtime.deviceSynchronize()
    t1 = perf_counter()
    param = pyqcu.QcuParam()
    param.lattice_size = latt_size
    grid = pyqcu.QcuParam()
    grid.lattice_size = grid_size
    pyqcu.dslashQcu(Mp.even_ptr, p.odd_ptr, U.data_ptr, param, 0)
    pyqcu.dslashQcu(Mp.odd_ptr, p.even_ptr, U.data_ptr, param, 1)
    # quda.dslashQuda(Mp.even_ptr, p.odd_ptr, dslash.invert_param,
    #                 QudaParity.QUDA_EVEN_PARITY)
    # quda.dslashQuda(Mp.odd_ptr, p.even_ptr, dslash.invert_param,
    #                 QudaParity.QUDA_ODD_PARITY)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    print(f'Quda dslash: {t2 - t1} sec')
    # then execute my code
    param = pyqcu.QcuParam()
    param.lattice_size = latt_size
    grid = pyqcu.QcuParam()
    grid.lattice_size = grid_size
    cp.cuda.runtime.deviceSynchronize()
    t1 = perf_counter()
    pyqcu.ncclDslashQcu(Mp1.even_ptr, p.odd_ptr, U.data_ptr, param, 0, grid)
    pyqcu.ncclDslashQcu(Mp1.odd_ptr, p.even_ptr, U.data_ptr, param, 1, grid)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    # pyqcu.testDslashQcu(Mp2.even_ptr, p.odd_ptr, U.data_ptr, param, 0)
    # pyqcu.testDslashQcu(Mp2.odd_ptr, p.even_ptr, U.data_ptr, param, 1)
    # cp.cuda.runtime.deviceSynchronize()
    print("######quda:Mp[0,0,0,0]:\n", Mp.lexico()[0, 0, 0, 0])
    print("######mpi:Mp1[0,0,0,0]:\n", Mp1.lexico()[0, 0, 0, 0])
    print("######quda:Mp[6,6,6,6]:\n", Mp.lexico()[6, 6, 6, 6])
    print("######mpi:Mp1[6,6,6,6]:\n", Mp1.lexico()[6, 6, 6, 6])
    print("######quda:Mp[-1,-1,-1,-1]:\n", Mp.lexico()[-1, -1, -1, -1])
    print("######mpi:Mp1[-1,-1,-1,-1]:\n", Mp1.lexico()[-1, -1, -1, -1])
    print("######quda:Mp[-6,-6,-6,-6]:\n", Mp.lexico()[-6, -6, -6, -6])
    print("######mpi:Mp1[-6,-6,-6,-6]:\n", Mp1.lexico()[-6, -6, -6, -6])
    # print("######test:Mp2[2,0,0,0]:\n",Mp2.lexico()[2,0,0,0])
    _ = np.where(
        np.sum(np.sum(np.abs(Mp1.data-Mp.data), axis=0), axis=0) > 1e-13)
    print("######T:", _[0], ",\n", len(_[0]))
    print("######Z:", _[1], ",\n", len(_[1]))
    print("######Y:", _[2], ",\n", len(_[2]))
    print("######X:", _[3], ",\n", len(_[3]))
    print(f'QCU dslash: {t2 - t1} sec')
    print('quda difference: ', cp.linalg.norm(
        Mp1.data - Mp.data) / cp.linalg.norm(Mp.data))


for i in range(0, 10):
    compare(i)
