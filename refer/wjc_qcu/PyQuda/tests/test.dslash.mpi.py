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

Nd, Ns, Nc = 4, 4, 3
# latt_size = [Lx, Ly, Lz, Lt]

#latt_size = [16, 16, 16, 32]
latt_size = [8, 8, 8, 16]
#latt_size = [8, 8, 8, 8]
#latt_size = [16, 16, 16, 32]
#latt_size = [8, 16, 16, 32]
#latt_size = [2, 2, 2, 2]
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
  #mpi.init(grid_size)
  from pyquda.mpi import comm, rank, size, grid, coord, gpuid
  
  p = LatticeFermion(latt_size, cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128))

  Mp = LatticeFermion(latt_size)
  Mp1 = LatticeFermion(latt_size)
  #dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)
  dslash = core.getDslash(latt_size, mass, 1e-9, 1000, xi_0, nu, coeff_t, coeff_r, multigrid=False, anti_periodic_t=False) #anti_periodic_t=False 
  #U = gauge_utils.gaussGauge(latt_size, round)
  U = gauge_utils.gaussGauge(latt_size, 0)
  dslash.loadGauge(U)
#  dslash.invert_param.dagger = 1

  cp.cuda.runtime.deviceSynchronize()
  
  t1 = perf_counter()
  quda.dslashQuda(Mp.even_ptr, p.odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
  quda.dslashQuda(Mp.odd_ptr, p.even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
  cp.cuda.runtime.deviceSynchronize()
  t2 = perf_counter()
  print(f"Quda dslash: {t2 - t1} sec")
  #print(f'=============Mp1 norm: , {cp.linalg.norm(Mp.data)}')
  

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
  t1 = perf_counter()

  #qcu.dslashQcu(Mp1.even_ptr, p.odd_ptr, U.data_ptr, param, 0)
  #qcu.dslashQcu(Mp1.odd_ptr, p.even_ptr, U.data_ptr, param, 1)

  qcu.dslashQcu(Mp1.even_ptr, p.even_ptr, U.data_ptr, param, 0) # last param: if dagger
  cp.cuda.runtime.deviceSynchronize()
  t2 = perf_counter()
  print(f"QCU dslash: {t2 - t1} sec")

  print(f'rank {rank} difference: , {cp.linalg.norm(Mp1.data - Mp.data) / cp.linalg.norm(Mp.data)}')
  #print(f'rank {rank} difference: , {cp.linalg.norm(Mp1.data - Mp.data)}')
  


  #print(f'rank = {rank}, res = {res[2,2,2,2]}')#, {cp.linalg.norm(res[1,1,2,2])}')
  #print(f'rank = {rank}, res1 = {res1[2,2,2,2]}')#, {cp.linalg.norm(res1[1,1,2,2])}')
  
  print('============================')
  #gauge = U.lexico()
  #print(gauge[0,0,0,0,0])
for test in range(0, 5):
    test_mpi(test)

def compare(round):
    # generate a vector p randomly
    p = LatticeFermion(latt_size, cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128))

    Mp = LatticeFermion(latt_size)
    Mp1 = LatticeFermion(latt_size)

    print("===============round ", round, "======================")
    # Set parameters in Dslash and use m=-3.5 to make kappa=1
    #dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)
    dslash = core.getDslash(latt_size, mass, 1e-9, 1000, xi_0, nu, coeff_t, coeff_r, multigrid=False,anti_periodic_t=False) #anti_periodic_t=False 
    # Generate gauge and then load it
    U = gauge_utils.gaussGauge(latt_size, round)
    dslash.loadGauge(U)

    cp.cuda.runtime.deviceSynchronize()

    t1 = perf_counter()
    quda.dslashQuda(Mp.even_ptr, p.odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
    quda.dslashQuda(Mp.odd_ptr, p.even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    print(f"Quda dslash: {t2 - t1} sec")

    # then execute my code
    param = qcu.QcuParam()
    param.lattice_size = latt_size

    cp.cuda.runtime.deviceSynchronize()
    t1 = perf_counter()
    qcu.dslashQcu(Mp1.even_ptr, p.odd_ptr, U.data_ptr, param, 0)
    qcu.dslashQcu(Mp1.odd_ptr, p.even_ptr, U.data_ptr, param, 1)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    print(f"QCU dslash: {t2 - t1} sec")

    print("difference: ", cp.linalg.norm(Mp1.data - Mp.data) / cp.linalg.norm(Mp.data))
    return U.lexico()



# for i in range(0, 5):
#     U = compare(i)



