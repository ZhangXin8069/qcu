# this file is modified from test.dslash.qcu.py
# 
import os
import sys
from time import perf_counter

import cupy as cp
import numpy as np

test_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.insert(0, os.path.join(test_dir, ".."))

from pyquda import init, core, quda, mpi, pyqcu as qcu
from pyquda.enum_quda import QudaParity
from pyquda.field import LatticeFermion
from pyquda.utils import gauge_utils

os.environ["QUDA_RESOURCE_PATH"] = ".cache"
# init()  # 在这里init会出问题

# Lx, Ly, Lz, Lt = 32, 32, 32, 64
#Lx, Ly, Lz, Lt = 16,16,16,32
Nd, Ns, Nc = 4, 4, 3

#latt_size = [16, 32, 32, 32]
#latt_size = [16, 32, 32, 64]
#latt_size = [16, 32, 32, 64]
latt_size = [16, 16, 32, 256]
#latt_size = [16, 16, 32, 64]
#latt_size = [16, 32, 32, 64]
#latt_size = [16, 16, 32, 64]
#grid_size = [1, 1, 2, 2]
grid_size = [1, 1, 1, 8]
#grid_size = [1, 1, 1, 1]
Lx, Ly, Lz, Lt = latt_size
Gx, Gy, Gz, Gt = grid_size
total_size = [Lx, Ly, Lz, Lt]
latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
print(f'vol = {Vol}')
xi_0, nu = 1, 1
mass=-3.5
coeff_r, coeff_t = 1,1


mpi.init(grid_size)




def test_mpi(round):

  from pyquda.mpi import comm, rank, size, grid, coord, gpuid
  
  p = LatticeFermion(latt_size, cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128))

  Mp = LatticeFermion(latt_size)
  Mp1 = LatticeFermion(latt_size)
  Mp2 = LatticeFermion(latt_size)
  x_vector = LatticeFermion(latt_size)
  #dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)

  dslash = core.getDslash(latt_size, mass, 1e-9, 1000, xi_0, nu, coeff_t, coeff_r, multigrid=False, anti_periodic_t=False) #anti_periodic_t=False 
  #U = gauge_utils.gaussGauge(latt_size, round)
  U = gauge_utils.gaussGauge(latt_size, 0)
  dslash.loadGauge(U)
  cp.cuda.runtime.deviceSynchronize()
  
  t1 = perf_counter()
  quda.dslashQuda(Mp.even_ptr, p.odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
  quda.dslashQuda(Mp.odd_ptr, p.even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
  cp.cuda.runtime.deviceSynchronize()
  t2 = perf_counter()
  print(f"Quda dslash: {t2 - t1} sec")

  #my code 
  param = qcu.QcuParam()
  grid = qcu.QcuGrid()
  param.lattice_size = latt_size
  grid.grid_size = grid_size
  qcu.initGridSize(grid, param, U.data_ptr, p.even_ptr, Mp1.even_ptr)
  qcu.loadQcuGauge(U.data_ptr, param)
  # then execute my code
  
  cp.cuda.runtime.deviceSynchronize()
  qcu.fullDslashQcu(Mp1.even_ptr, p.even_ptr, U.data_ptr, param, 0)  # full Dslash ----> Mp1


  t1 = perf_counter()
  qcu.cg_inverter(Mp1.even_ptr, x_vector.even_ptr, U.data_ptr, param) # Dslash x_vector = Mp1, get x_vector
  t2 = perf_counter()
  qcu.fullDslashQcu(Mp2.even_ptr, x_vector.even_ptr, U.data_ptr, param, 0) # Dslash x_vector--->Mp2
  print(f'rank {rank} my x and x difference: , {cp.linalg.norm(Mp1.data - Mp2.data) / cp.linalg.norm(Mp2.data)}, takes {t2 - t1} sec')
  print(f'total_size = {total_size}, every process get latt_size = {latt_size}, mpi_size = {grid_size}')
  
  
  print('============================')

for test in range(0, 10):
    test_mpi(test)


