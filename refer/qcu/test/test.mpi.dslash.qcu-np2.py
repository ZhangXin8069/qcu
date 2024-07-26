# this file is modified from test.dslash.qcu.py
#
from pyquda.utils import gauge_utils
from pyquda.field import LatticeFermion
from pyquda.enum_quda import QudaParity
from pyquda import init, core, quda, mpi, pyqcu as qcu
import os
import sys
from time import perf_counter

import cupy as cp
import numpy as np

test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(test_dir, ".."))


os.environ["QUDA_RESOURCE_PATH"] = ".cache"

Nd, Ns, Nc = 4, 4, 3
latt_size = [16, 16, 16, 32]
# latt_size = [8, 8, 8, 16]
# latt_size = [8, 4, 4, 4]
# grid_size = [1, 1, 2, 2]
grid_size = [2, 1, 1, 1]
# grid_size = [1, 1, 1, 16]
Lx, Ly, Lz, Lt = latt_size
Gx, Gy, Gz, Gt = grid_size
latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
# print(f'vol = {Vol}')
xi_0, nu = 1, 1
mass = 0
# coeff_r, coeff_t = 1,1
coeff_r, coeff_t = 0, 0


mpi.init(grid_size)


def test_mpi(round):
    # generate a vector p randomly
    p = LatticeFermion(latt_size, cp.random.randn(
        Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128))
    Mp = LatticeFermion(latt_size)
    Mp1 = LatticeFermion(latt_size)

    print("===============round ", round, "======================")

    # Set parameters in Dslash and use m=-3.5 to make kappa=1
    dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)
    # Generate gauge and then load it
    U = gauge_utils.gaussGauge(latt_size, round)
    dslash.loadGauge(U)

    cp.cuda.runtime.deviceSynchronize()
    t1 = perf_counter()
    quda.dslashQuda(Mp.even_ptr, p.odd_ptr, dslash.invert_param,
                    QudaParity.QUDA_EVEN_PARITY)
    quda.dslashQuda(Mp.odd_ptr, p.even_ptr, dslash.invert_param,
                    QudaParity.QUDA_ODD_PARITY)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    print(f"Quda dslash: {t2 - t1} sec")

    # then execute my code
    param = qcu.QcuParam()
    param.lattice_size = latt_size
    # U.data = cp.ascontiguousarray(U.data[:, :, :, :, :, :, :2, :])

    cp.cuda.runtime.deviceSynchronize()
    t1 = perf_counter()
    qcu.dslashQcu(Mp1.even_ptr, p.odd_ptr, U.data_ptr, param, 0)
    qcu.dslashQcu(Mp1.odd_ptr, p.even_ptr, U.data_ptr, param, 1)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    print(f"QCU dslash: {t2 - t1} sec")

    print("difference: ", cp.linalg.norm(
        Mp1.data - Mp.data) / cp.linalg.norm(Mp.data))


for test in range(0, 10):
    test_mpi(test)

