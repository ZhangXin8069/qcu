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
os.environ["QUDA_RESOURCE_PATH"] = ".cache"
latt_size = [16, 32, 32, 64]
grid_size = [1, 1, 1, 1]
Lx, Ly, Lz, Lt = latt_size
Nd, Ns, Nc = 4, 4, 3
Gx, Gy, Gz, Gt = grid_size
latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
mpi.init(grid_size)


def dslash_qcu(Mp, p, U, param, kappa):
    pyqcu.dslashQcu(Mp.even_ptr, p.odd_ptr, U.data_ptr, param, 0)
    pyqcu.dslashQcu(Mp.odd_ptr, Mp.even_ptr, U.data_ptr, param, 1)
    Mp = p - kappa*kappa*Mp


x_origion = cp.random.randn(
    Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128)*1  # ?
print("x_origion = ", x_origion.data[0, 0, 0, 0, 0, 0, :])
r0 = cp.zeros((Lt, Lz, Ly, Lx//2, Ns, Nc), cp.complex128)
r1 = cp.zeros((Lt, Lz, Ly, Lx//2, Ns, Nc), cp.complex128)
t = cp.zeros((Lt, Lz, Ly, Lx//2, Ns, Nc), cp.complex128)
p = cp.zeros((Lt, Lz, Ly, Lx//2, Ns, Nc), cp.complex128)
tmp = cp.zeros((Lt, Lz, Ly, Lx//2, Ns, Nc), cp.complex128)
Ap = cp.zeros((Lt, Lz, Ly, Lx//2, Ns, Nc), cp.complex128)
b = cp.zeros((Lt, Lz, Ly, Lx//2, Ns, Nc), cp.complex128)
param = pyqcu.QcuParam()
param.lattice_size = latt_size
dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)
kappa = 0.125
U = gauge_utils.gaussGauge(latt_size, 0)
dslash.loadGauge(U)
pyqcu.dslashQcu(tmp.even_ptr, x_origion.odd_ptr, U.data_ptr, param, 0)
pyqcu.dslashQcu(tmp.odd_ptr, x_origion.even_ptr, U.data_ptr, param, 1)
b.data[:] = x_origion.data[:] - kappa*tmp.data[:]
pyqcu.dslashQcu(tmp.odd_ptr, b.even_ptr, U.data_ptr, param, 1)
b += kappa*tmp
dslash_qcu(tmp, x_origion, U, param, kappa)
x0 = cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128)*1
x = LatticeFermion(latt_size, x0)
cp.cuda.runtime.deviceSynchronize()
t1 = perf_counter()
dslash_qcu(tmp, x, U, param, kappa)
r = b - tmp
r0 = r
p = r
turns = 0
for i in range(1, 300):
    norm_r = cp.linalg.norm(r)
    dslash_qcu(tmp, p, U, param, kappa)
    alpha = cp.inner(r0.flatten().conjugate(), r.flatten(
    ))/cp.inner(r0.flatten().conjugate(), tmp.flatten())
    x = x + alpha*p
    r1 = r - alpha*tmp
    Ap = tmp
    dslash_qcu(tmp, r, U, param, kappa)
    t = tmp
    omega = cp.inner(t.flatten().conjugate(), r.flatten(
    ))/cp.inner(t.flatten().conjugate(), t.flatten())
    x = x + omega*r1
    dslash_qcu(tmp, r1, U, param, kappa)
    r1 = r1 - omega*tmp
    beta = cp.inner(r1.flatten().conjugate(), r1.flatten(
    ))/cp.inner(r.flatten().conjugate(), r.flatten())
    p = r1 + (alpha*beta)/omega*p - (alpha*beta)*Ap
    r = r1
    dslash_qcu(tmp, x, U, param, kappa)
    cp.cuda.runtime.deviceSynchronize()
    if (norm_r < 10e-16 or cp.isnan(norm_r)):
        turns = i
        break
print('difference: ', cp.linalg.norm(
    x - x_origion) / cp.linalg.norm(x_origion))
tmp = x_origion - kappa*tmp
print("turns = ", turns, '\n')
t2 = perf_counter()
print(f'Quda dslash: {t2 - t1} sec')
