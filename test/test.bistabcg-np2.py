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
latt_size = [32, 32, 32, 32]
# latt_size = [32, 32, 32, 32]
# latt_size = [8, 8, 8, 8]
# latt_size = [24, 24, 24, 72]
grid_size = [2, 1, 1, 1]
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
print(f"single latt size = {latt_size}")
# set
# p = LatticeFermion(latt_size, cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128))
p = LatticeFermion(latt_size, cp.ones(
    [Lt, Lz, Ly, Lx, Ns, Nc * 2]).view(cp.complex128))
qcu_p = LatticeFermion(latt_size)
quda_p = LatticeFermion(latt_size)
qcu_x = LatticeFermion(latt_size)
quda_x = LatticeFermion(latt_size)
dslash = core.getDslash(
    latt_size,
    mass,
    1e-9,
    1000,
    xi_0,
    nu,
    coeff_t,
    coeff_r,
    multigrid=False,
    anti_periodic_t=False,
)
# dslash.invert_param.inv_type = 0  # QUDA_CG_INVERTER
# dslash.invert_param.inv_type = 1  # QUDA_BICGSTAB_INVERTER
# dslash.invert_param.inv_type = 13  # QUDA_BICGSTABL_INVERTER
U = gauge_utils.gaussGauge(latt_size, 0)
dslash.loadGauge(U)


def compare(round):
    # quda
    cp.cuda.runtime.deviceSynchronize()
    if rank == 0:
        print("================quda=================")
    t1 = perf_counter()
    quda.invertQuda(quda_x.data_ptr, p.data_ptr, dslash.invert_param)
    # D*x=p, to get quda_x
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    quda.MatQuda(quda_p.data_ptr, quda_x.data_ptr, dslash.invert_param)
    # quda_p=D*quda_x
    cp.cuda.runtime.deviceSynchronize()
    print(
        f"rank {rank} quda x and x difference: , {cp.linalg.norm(quda_p.data - p.data) / cp.linalg.norm(quda_p.data)}, takes {t2 - t1} sec, norm_quda_x = {cp.linalg.norm(quda_x.data)}"
    )
    print(f"quda rank {rank} takes {t2 - t1} sec")
    # float test
    t1 = perf_counter()
    print(U.data_ptr)
    print(type(U.data))
    print(U.data.dtype)
    U.data = U.data.astype(cp.complex64)
    print(U.data_ptr)
    print(type(U.data))
    print(U.data.dtype)
    print(p.data_ptr)
    print(type(p.data))
    print(p.data.dtype)
    p.data = p.data.astype(cp.complex64)
    print(p.data_ptr)
    print(type(p.data))
    print(p.data.dtype)
    print(quda_x.data_ptr)
    print(type(quda_x.data))
    print(quda_x.data.dtype)
    quda_x.data = quda_x.data.astype(cp.complex64)
    print(quda_x.data_ptr)
    print(type(quda_x.data))
    print(quda_x.data.dtype)
    print(qcu_x.data_ptr)
    print(type(qcu_x.data))
    print(qcu_x.data.dtype)
    qcu_x.data = qcu_x.data.astype(cp.complex64)
    print(qcu_x.data_ptr)
    print(type(qcu_x.data))
    print(qcu_x.data.dtype)
    t2 = perf_counter()
    print(f'turn data to float: {t2 - t1} sec')
    U.data.astype(cp.complex64).tofile("wilson-bistabcg-gauge_-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-f.bin".format(
        Lx, Ly, Lz, Lt, Lx*Ly*Lz*Lt, Gx, Gy, Gz, Gt, 0, mpi.rank, mpi.size, 0))
    p.data.tofile("wilson-bistabcg-fermion-in_-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-f.bin".format(
        Lx, Ly, Lz, Lt, Lx*Ly*Lz*Lt, Gx, Gy, Gz, Gt, 0, mpi.rank, mpi.size, 0))
    quda_x.data.tofile("wilson-bistabcg-fermion-out_-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-f.bin".format(
        Lx, Ly, Lz, Lt, Lx*Ly*Lz*Lt, Gx, Gy, Gz, Gt, 0, mpi.rank, mpi.size, 0))
    ######
    # qcu
    param = qcu.QcuParam()
    param.lattice_size = latt_size
    grid = qcu.QcuParam()
    grid.lattice_size = grid_size
    # qcu_x.data = quda_x.data.copy()
    cp.cuda.runtime.deviceSynchronize()
    if rank == 0:
        print("===============qcu==================")
    t1 = perf_counter()
    qcu.applyBistabCgQcu(qcu_x.data_ptr, p.data_ptr, U.data_ptr, param, grid)
    # qcu.applyBistabCgQcu(qcu_x.data_ptr,
    #                     quda_x.data_ptr, U.data_ptr, param, grid)
    # D*x=p, to get qcu_x
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    quda.MatQuda(qcu_p.data_ptr, qcu_x.data_ptr, dslash.invert_param)
    # qcu_p=D*qcu_x
    print(
        f"rank {rank} my x and x difference: , {cp.linalg.norm(qcu_p.data - p.data) / cp.linalg.norm(qcu_p.data)}, takes {t2 - t1} sec, my_x_norm = {cp.linalg.norm(qcu_x.data)}"
    )
    print(f"qcu rank {rank} takes {t2 - t1} sec")
    print("============================")
    print('quda difference: ', cp.linalg.norm(
        qcu_x.data - quda_x.data) / cp.linalg.norm(quda_x.data))


for i in range(0, 1):
    compare(i)
