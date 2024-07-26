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
import matplotlib.pyplot as plt

test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(test_dir, ".."))


os.environ["QUDA_RESOURCE_PATH"] = ".cache"

Nd, Ns, Nc = 4, 4, 3

# latt_size = [8,8,8,8]
latt_size = [16, 16, 16, 32]

grid_size = [1, 1, 1, 1]
Lx, Ly, Lz, Lt = latt_size
Gx, Gy, Gz, Gt = grid_size
latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
print(f'vol = {Vol}')
xi_0, nu = 1, 1
# mass = -3.5
mass = 0
# coeff_r, coeff_t = 1,1
coeff_r, coeff_t = 0, 0  # wilson


mpi.init(grid_size)

param = qcu.QcuParam()
grid = qcu.QcuGrid()
param.lattice_size = latt_size
grid.grid_size = grid_size

half_prec = 0
float_prec = 1
double_prec = 2


def test_mpi(round, my_m_input):
    from pyquda.mpi import comm, rank, size, grid, coord, gpuid
    p_mrhs = [LatticeFermion(latt_size, cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128))
              for i in range(my_m_input)]

    quda_Mp_mrhs = [LatticeFermion(latt_size) for i in range(my_m_input)]
    qcu_Mp_mrhs = [LatticeFermion(latt_size) for i in range(my_m_input)]

    dslash = core.getDslash(latt_size, mass, 1e-9, 1000, xi_0, nu,
                            coeff_t, coeff_r, multigrid=False, anti_periodic_t=False)
    U = gauge_utils.gaussGauge(latt_size, 0)

    dslash.loadGauge(U)
    cp.cuda.runtime.deviceSynchronize()

    t1 = perf_counter()
    for i in range(my_m_input):
        quda.dslashQuda(quda_Mp_mrhs[i].even_ptr, p_mrhs[i].odd_ptr,
                        dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
        quda.dslashQuda(quda_Mp_mrhs[i].odd_ptr, p_mrhs[i].even_ptr,
                        dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    quda_dslash_time = t2 - t1

    # my code
    qcu.loadQcuGauge(U.data_ptr, 2)		# 2---double 1--float 0---half
    qcu.getDslash(0, mass)  # 0----WILSON
    cp.cuda.runtime.deviceSynchronize()

    t1 = perf_counter()
    for i in range(my_m_input):
        qcu.pushBackFermions(qcu_Mp_mrhs[i].even_ptr, p_mrhs[i].odd_ptr)
    qcu.start_dslash(0, 0)  # param1 : parity  param2: dagger

    for i in range(my_m_input):
        qcu.pushBackFermions(qcu_Mp_mrhs[i].odd_ptr, p_mrhs[i].even_ptr)
    qcu.start_dslash(1, 0)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    qcu_dslash_time = t2 - t1

    print(
        f"Quda dslash: {quda_dslash_time}sec \nQcu dslash:  {qcu_dslash_time} sec")
    # for i in range(my_m_input) :
    #   print(f'rank {rank}, Mp[{i}] difference: , \
    #         {cp.linalg.norm(quda_Mp_mrhs[i].data - qcu_Mp_mrhs[i].data) / cp.linalg.norm(quda_Mp_mrhs[i].data)}')

    average_difference = cp.sum(cp.array([cp.linalg.norm(quda_Mp_mrhs[i].data - qcu_Mp_mrhs[i].data) / cp.linalg.norm(quda_Mp_mrhs[i].data)
                                          for i in range(my_m_input)])) / my_m_input
    print(f'rank {rank}, average difference: , {average_difference}')
    print('===============================')
    return quda_dslash_time, qcu_dslash_time


def test_dslash(my_n_color, my_m_input, input_prec, dslash_prec, quda_average_time, qcu_average_time) -> int:
    qcu.initGridSize(grid, param, my_n_color,
                     my_m_input, input_prec, dslash_prec)

    total_quda_time = 0
    total_qcu_time = 0

    iteration = 10
    for i in range(iteration):
        quda_time, qcu_time = test_mpi(i, my_m_input)
        total_quda_time += quda_time
        total_qcu_time += qcu_time
    quda_average_time.append(total_quda_time / iteration)
    qcu_average_time.append(total_qcu_time / iteration)

    qcu.finalizeQcu()
    cp.cuda.runtime.deviceSynchronize()


if __name__ == '__main__':
    max_input = 24
    my_n_color = 3

    quda_average_time = []
    qcu_average_time = []

    for my_m_input in range(1, max_input+1):
        test_dslash(my_n_color, my_m_input, input_prec=double_prec, dslash_prec=half_prec,
                    quda_average_time=quda_average_time, qcu_average_time=qcu_average_time)
    # test_dslash(my_n_color, 1, double_prec, double_prec, quda_average_time, qcu_average_time)

    print(f'quda_average_time: {quda_average_time}')
    print(f'qcu_average_time: {qcu_average_time}')

    x = np.arange(1, max_input+1, 1)
    quda_per_rhs = quda_average_time / x
    qcu_per_rhs = qcu_average_time / x
    plt.plot(x, quda_per_rhs, label='quda', marker='o')
    plt.plot(x, qcu_per_rhs, linestyle='--', label='qcu', marker='o')
    plt.title(f'average dslash time per rhs, latt size = {latt_size}')
    plt.xlabel('m_input')
    plt.ylabel('time')
    plt.legend()
    plt.show()
    plt.savefig('dslash_result_per_rhs.png')

    plt.clf()
    plt.plot(x, quda_average_time, label='quda', marker='o')
    plt.plot(x, qcu_average_time, linestyle='--', label='qcu', marker='o')
    plt.title(f'average dslash time, latt size = {latt_size}')
    plt.xlabel('m_input')
    plt.ylabel('time')
    plt.legend()
    plt.show()
    plt.savefig('dslash_result.png')
