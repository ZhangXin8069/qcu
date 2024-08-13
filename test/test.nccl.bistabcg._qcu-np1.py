from pyquda.mpi import comm, rank, size, grid, coord, gpuid
from pyquda.utils import gauge_utils
from pyquda.field import LatticeFermion
from pyquda.enum_quda import QudaParity
from pyquda import init, core, quda, mpi, qcu as qcu
import os
import sys
from time import perf_counter
import cupy as cp
import numpy as np
test_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["QUDA_RESOURCE_PATH"] = ".cache"
Nd, Ns, Nc = 4, 4, 3
latt_size = [16, 16, 16, 32]
grid_size = [1, 1, 1, 1]
Lx, Ly, Lz, Lt = latt_size
Gx, Gy, Gz, Gt = grid_size
latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
xi_0, nu = 1, 1
mass = 0
# coeff_r, coeff_t = 1,1
coeff_r, coeff_t = 0, 0
mpi.init(grid_size)
print(f'single latt size = {latt_size}')
# set
p = LatticeFermion(latt_size, cp.random.randn(
    Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128))
# p = LatticeFermion(latt_size, cp.ones(
#     [Lt, Lz, Ly, Lx, Ns, Nc * 2]).view(cp.complex128))
x = LatticeFermion(latt_size, cp.random.randn(
    Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128))
qcu_p = LatticeFermion(latt_size)
quda_p = LatticeFermion(latt_size)
qcu_x = LatticeFermion(latt_size)
quda_x = LatticeFermion(latt_size)
dslash = core.getDslash(latt_size, mass, 1e-9, 1000, xi_0, nu, coeff_t,
                        coeff_r, multigrid=False, anti_periodic_t=False)
U = gauge_utils.gaussGauge(latt_size, 0)
qcu_U = gauge_utils.gaussGauge(latt_size, 0)
dslash.loadGauge(U)


def compare(round):
    # quda
    cp.cuda.runtime.deviceSynchronize()
    if rank == 0:
        print('================quda=================')
    t1 = perf_counter()
    quda.invertQuda(quda_x.data_ptr, p.data_ptr, dslash.invert_param)
    # D*x=p, to get quda_x
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    quda.MatQuda(quda_p.data_ptr, quda_x.data_ptr, dslash.invert_param)
    # quda_p=D*quda_x
    cp.cuda.runtime.deviceSynchronize()
    print(f'rank {rank} quda x and x difference: , {cp.linalg.norm(quda_p.data - p.data) / cp.linalg.norm(quda_p.data)}, takes {t2 - t1} sec, norm_quda_x = {cp.linalg.norm(quda_x.data)}')
    print(f'quda rank {rank} takes {t2 - t1} sec')
    # qcu
    param = qcu.QcuParam()
    param.lattice_size = latt_size
    grid = qcu.QcuParam()
    grid.lattice_size = grid_size
    cp.cuda.runtime.deviceSynchronize()
    if rank == 0:
        print('===============qcu==================')
    qcu_x.data[:] = x.data
    qcu_p.data[:] = p.data
    qcu_U.data[:] = U.data
    t1 = perf_counter()
    # qcu.ncclBistabCgQcu(qcu_x.data_ptr, p.data_ptr, U.data_ptr, param, grid)
    # qcu.ncclBistabCgQcu(qcu_x.data_ptr,
    #                     quda_x.data_ptr, U.data_ptr, param, grid)
    # D*x=p, to get qcu_x
    # test
    qcu.ncclBistabCgQcu(qcu_x.data_ptr, qcu_p.data_ptr,
                        qcu_U.data_ptr, param, grid)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    # quda.MatQuda(qcu_p.data_ptr, qcu_x.data_ptr, dslash.invert_param)
    # qcu_p=D*qcu_x
    print(f'rank {rank} my x and x difference: {cp.linalg.norm(qcu_x.data - x.data) / cp.linalg.norm(x.data)}, takes {t2 - t1} sec, my_x_norm = {cp.linalg.norm(qcu_x.data)}, x_norm = {cp.linalg.norm(x.data)}')
    print(f'rank {rank} my p and p difference: {cp.linalg.norm(qcu_p.data - p.data) / cp.linalg.norm(p.data)}, takes {t2 - t1} sec, my_p_norm = {cp.linalg.norm(qcu_p.data)}, p_norm = {cp.linalg.norm(p.data)}')
    print(f'rank {rank} my U and U difference: {cp.linalg.norm(qcu_U.data - U.data) / cp.linalg.norm(U.data)}, takes {t2 - t1} sec, my_U_norm = {cp.linalg.norm(qcu_U.data)}, U_norm = {cp.linalg.norm(U.data)}')
    # print(f'rank {rank} my x and x difference: , {cp.linalg.norm(qcu_p.data - p.data) / cp.linalg.norm(qcu_p.data)}, takes {t2 - t1} sec, my_x_norm = {cp.linalg.norm(qcu_x.data)}')
    print(x.data.shape)
    print(x.data[0, 5, 4, 3, 2, 1, 0])
    print(qcu_x.data[0].reshape(4, 3,  32, 16, 16, 8)[1, 0, 5, 4, 3, 2])
    print(x.data[0, 6, 5, 4, 3, 2, 1])
    print(qcu_x.data[0].reshape(4, 3,  32, 16, 16, 8)[2, 1, 6, 5, 4, 3])
    print(p.data.shape)
    print(p.data[0, 5, 4, 3, 2, 1, 0])
    print(qcu_p.data[0].reshape(4, 3,  32, 16, 16, 8)[1, 0, 5, 4, 3, 2])
    print(p.data[0, 6, 5, 4, 3, 2, 1])
    print(qcu_p.data[0].reshape(4, 3,  32, 16, 16, 8)[2, 1, 6, 5, 4, 3])
    print(U.data.shape)
    print("####")
    print(U.data[1, 0, 5, 4, 3, 2, 1, 0])
    print(qcu_U.data.reshape(
        3, 3, 4, 2, 32, 16, 16, 8)[1, 0, 1, 0, 5, 4, 3, 2])
    print(U.data[2, 1, 6, 5, 4, 3, 2, 1])
    print(qcu_U.data.reshape(
        3, 3, 4, 2, 32, 16, 16, 8)[2, 1, 2, 1, 6, 5, 4, 3])
    print(f'qcu rank {rank} takes {t2 - t1} sec')
    print('============================')


for i in range(0, 10):
    compare(i)