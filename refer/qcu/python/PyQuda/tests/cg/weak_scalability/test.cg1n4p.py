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
latt_size = [16, 16, 32, 128]
#latt_size = [8, 8, 8, 16]
#latt_size = [8, 4, 4, 4]
grid_size = [1, 1, 1, 4]
#grid_size = [2, 2, 2, 2]
#grid_size = [1, 1, 1, 16]
Lx, Ly, Lz, Lt = latt_size
Gx, Gy, Gz, Gt = grid_size
latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
#print(f'vol = {Vol}')
xi_0, nu = 1, 1
mass=0
#coeff_r, coeff_t = 1,1
coeff_r, coeff_t = 0,0


mpi.init(grid_size)




def test_mpi(round):

  from pyquda.mpi import comm, rank, size, grid, coord, gpuid
  print(f'single latt size = {latt_size}')  
  #p = LatticeFermion(latt_size, cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128))
  p = LatticeFermion(latt_size, cp.ones([Lt, Lz, Ly, Lx, Ns, Nc * 2]).view(cp.complex128))
  print(f'norm(p) = {cp.linalg.norm(p.data)}')
  print(f'norm(p[0]) = {cp.linalg.norm(p.data[0])}')
  print(f'norm(p[1]) = {cp.linalg.norm(p.data[1])}')

 
  Mp1 = LatticeFermion(latt_size)
  Mp2 = LatticeFermion(latt_size)
  x_vector = LatticeFermion(latt_size)
  quda_x = LatticeFermion(latt_size)
  #dslash = core.getDslash(latt_size, mass, 0, 0, anti_periodic_t=False)

  dslash = core.getDslash(latt_size, mass, 1e-9, 1000, xi_0, nu, coeff_t, coeff_r, multigrid=False, anti_periodic_t=False) #anti_periodic_t=False 
  #dslash = core.getDslash(latt_size, mass, 1e-9, 2, xi_0, nu, coeff_t, coeff_r, multigrid=False, anti_periodic_t=False) #anti_periodic_t=False 
  #U = gauge_utils.gaussGauge(latt_size, round)
  U = gauge_utils.gaussGauge(latt_size, 0)
  dslash.loadGauge(U)
  
  
  # quda_x = dslash.invert(Mp1)
  cp.cuda.runtime.deviceSynchronize()  
  t1 = perf_counter()
  if rank == 0:
    print('================quda=================')
  #quda_x = dslash.invert(Mp1)
  #quda_x = dslash.invert(p)
  quda.invertQuda(quda_x.data_ptr, p.data_ptr, dslash.invert_param)
  cp.cuda.runtime.deviceSynchronize()  
  t2 = perf_counter()
  quda.MatQuda(Mp2.data_ptr, quda_x.data_ptr, dslash.invert_param)
  cp.cuda.runtime.deviceSynchronize()
  print(f'rank {rank} quda x and x difference: , {cp.linalg.norm(Mp2.data - p.data) / cp.linalg.norm(Mp2.data)}, takes {t2 - t1} sec, norm_quda_x = {cp.linalg.norm(quda_x.data)}')
  comm.Barrier()  
 
  #my code 
  param = qcu.QcuParam()
  grid = qcu.QcuGrid()
  param.lattice_size = latt_size
  grid.grid_size = grid_size
  qcu.initGridSize(grid, param, U.data_ptr, p.even_ptr, Mp1.even_ptr)
  qcu.loadQcuGauge(U.data_ptr, param)
  # then execute my code
  
  #qcu.fullDslashQcu(Mp1.even_ptr, p.even_ptr, U.data_ptr, param, 0)  # full Dslash ----> Mp1


  #qcu.cg_inverter(Mp1.even_ptr, x_vector.even_ptr, U.data_ptr, param) # Dslash x_vector = Mp1, get x_vector
  cp.cuda.runtime.deviceSynchronize()
  comm.Barrier()  
  if rank == 0: 
    print('===============qcu==================')
  t1 = perf_counter()
  qcu.cg_inverter(x_vector.even_ptr, p.even_ptr, U.data_ptr, param, 1e-9, 0.125) # Dslash x_vector = Mp1, get x_vector
  cp.cuda.runtime.deviceSynchronize()
  t2 = perf_counter()
  quda.MatQuda(Mp1.data_ptr, x_vector.data_ptr, dslash.invert_param)
  
  print(f'rank {rank} qcu x and x difference: , {cp.linalg.norm(Mp1.data - p.data) / cp.linalg.norm(Mp1.data)}, takes {t2 - t1} sec, my_x_norm = {cp.linalg.norm(x_vector.data)}')
  if rank == 0: 
    print('============================')

for test in range(0, 10):
    test_mpi(test)



