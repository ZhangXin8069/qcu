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

Nd, Ns, Nc = 4, 4, 3

latt_size = [8, 8, 8, 8]
grid_size = [1, 1, 1, 1]
Lx, Ly, Lz, Lt = latt_size
Gx, Gy, Gz, Gt = grid_size
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
  #dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)   #wilson dslash

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
#  cp.cuda.runtime.deviceSynchronize()
  qcu.initGridSize(grid, param, U.data_ptr, p.even_ptr, Mp1.even_ptr)

  # then execute my code
  #param = qcu.QcuParam()
  #param.lattice_size = latt_size
  
  cp.cuda.runtime.deviceSynchronize()

  print(f'===============================================')
  qcu.dslashQcu(Mp1.even_ptr, p.even_ptr, U.data_ptr, param, 1)
  #print(f'rank {rank} difference: , {cp.linalg.norm(Mp1.data - Mp.data)}')

  qcu.cg_inverter(Mp1.even_ptr, p.even_ptr, U.data_ptr, param)


  
  
  print('================================================')

for test in range(0, 10):
    test_mpi(test)





def quda_cg (b_vec, new_b_vec, U, dslash):
  # assume space allocated
  from pyquda.mpi import comm, rank, size, grid, coord, gpuid
  
  #dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)

  #dslash = core.getDslash(latt_size, mass, 1e-9, 1000, xi_0, nu, coeff_t, coeff_r, multigrid=False, anti_periodic_t=False) #anti_periodic_t=False 
  #U = gauge_utils.gaussGauge(latt_size, 0)
  #dslash.loadGauge(U)

  dslash.invert_param.dagger = 1

  quda.dslashQuda(new_b.even_ptr, b.odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
  quda.dslashQuda(new_b.odd_ptr, b.even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)

  new_b = new_b - b     # src - kappa src
  
  r_vec = LatticeFermion(latt_size)
  p_vec = LatticeFermion(latt_size)
  temp1 = LatticeFermion(latt_size)
  temp2 = LatticeFermion(latt_size)
  x_vec = LatticeFermion(latt_size)
  print(f'initialize: the initial value of x_vec is {cp.linalg.norm(x_vec.data}')


  # Ax ----> temp
  dslash.invert_param.dagger = 0
  quda.dslashQuda(temp1.even_ptr, x_vec.odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
  quda.dslashQuda(temp1.odd_ptr, x_vec.even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
  temp1.data = temp1.data - x_vec.data   # src - kappa temp
  dslash.invert_param.dagger = 1
  quda.dslashQuda(temp2.even_ptr, temp1.odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
  quda.dslashQuda(temp2.odd_ptr, temp1.even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
  temp2.data = temp2.data - temp1.data   # src - kappa temp


  
  print(f"Quda dslash: {t2 - t1} sec")


