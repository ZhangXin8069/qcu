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
latt_size = [16, 16, 16, 32]
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
    Mp.data[1, :] = p.data[1, :] - kappa*kappa*Mp.data[1, :]


x1 = cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128)*1
x_ans = LatticeFermion(latt_size, x1)
print("x_ans = ", x_ans.data[0, 0, 0, 0, 0, 0, :])
r1 = cp.zeros((2, Lt, Lz, Ly, Lx//2, Ns, Nc), cp.complex128)
r = LatticeFermion(latt_size, r1)
r2 = cp.zeros((2, Lt, Lz, Ly, Lx//2, Ns, Nc), cp.complex128)
r0 = LatticeFermion(latt_size, r2)
t0 = cp.zeros((2, Lt, Lz, Ly, Lx//2, Ns, Nc), cp.complex128)
t = LatticeFermion(latt_size, t0)
r_10 = cp.zeros((2, Lt, Lz, Ly, Lx//2, Ns, Nc), cp.complex128)
r_1 = LatticeFermion(latt_size, r_10)
p0 = cp.zeros((2, Lt, Lz, Ly, Lx//2, Ns, Nc), cp.complex128)
p = LatticeFermion(latt_size, p0)
tmp0 = cp.zeros((2, Lt, Lz, Ly, Lx//2, Ns, Nc), cp.complex128)
tmp = LatticeFermion(latt_size, tmp0)
tmp10 = cp.zeros((2, Lt, Lz, Ly, Lx//2, Ns, Nc), cp.complex128)
tmp1 = LatticeFermion(latt_size, tmp10)
Ap0 = cp.zeros((2, Lt, Lz, Ly, Lx//2, Ns, Nc), cp.complex128)
Ap = LatticeFermion(latt_size, Ap0)
b0 = cp.zeros((2, Lt, Lz, Ly, Lx//2, Ns, Nc), cp.complex128)
b = LatticeFermion(latt_size, b0)
param = pyqcu.QcuParam()
param.lattice_size = latt_size
dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)
kappa = 0.125
U = gauge_utils.gaussGauge(latt_size, 0)
dslash.loadGauge(U)
pyqcu.dslashQcu(tmp.even_ptr, x_ans.odd_ptr, U.data_ptr, param, 0)
pyqcu.dslashQcu(tmp.odd_ptr, x_ans.even_ptr, U.data_ptr, param, 1)
b.data[:] = x_ans.data[:] - kappa*tmp.data[:]
pyqcu.dslashQcu(tmp.odd_ptr, b.even_ptr, U.data_ptr, param, 1)
b.data[1, :] += kappa*tmp.data[1, :]
dslash_qcu(tmp, x_ans, U, param, kappa)
x0 = cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128)*1
x = LatticeFermion(latt_size, x0)
cp.cuda.runtime.deviceSynchronize()
dslash_qcu(tmp, x, U, param, kappa)
r.data[1, :] = b.data[1, :] - tmp.data[1, :]
r0.data[1, :] = r.data[1, :]
p.data[1, :] = r.data[1, :]
t1 = perf_counter()
for i in range(1, 3000):
    norm_r = cp.linalg.norm(r.data[1, :])
    dslash_qcu(tmp, p, U, param, kappa)
    alpha = cp.inner(r0.data[1, :].flatten().conjugate(), r.data[1, :].flatten(
    ))/cp.inner(r0.data[1, :].flatten().conjugate(), tmp.data[1, :].flatten())
    x.data[1, :] = x.data[1, :] + alpha*p.data[1, :]
    r_1.data[1, :] = r.data[1, :] - alpha*tmp.data[1, :]
    Ap.data[1, :] = tmp.data[1, :]
    dslash_qcu(tmp, r, U, param, kappa)
    t.data[1, :] = tmp.data[1, :]
    omega = cp.inner(t.data[1, :].flatten().conjugate(), r.data[1, :].flatten(
    ))/cp.inner(t.data[1, :].flatten().conjugate(), t.data[1, :].flatten())
    x.data[1, :] = x.data[1, :] + omega*r_1.data[1, :]
    dslash_qcu(tmp, r_1, U, param, kappa)
    r_1.data[1, :] = r_1.data[1, :] - omega*tmp.data[1, :]
    beta = cp.inner(r_1.data[1, :].flatten().conjugate(), r_1.data[1, :].flatten(
    ))/cp.inner(r.data[1, :].flatten().conjugate(), r.data[1, :].flatten())
    p.data[1, :] = r_1.data[1, :] + \
        (alpha*beta)/omega*p.data[1, :] - (alpha*beta)*Ap.data[1, :]
    r.data[1, :] = r_1.data[1, :]
    dslash_qcu(tmp, x, U, param, kappa)
    cp.cuda.runtime.deviceSynchronize()
    if (norm_r < 10e-16 or cp.isnan(norm_r)):
        print("turns = ", i, '\n')
        break
print("x = ", x.data[0, 0, 0, 0, 0, 0, :])
print('difference: ', cp.linalg.norm(
    x.data[1, :] - x_ans.data[1, :]) / cp.linalg.norm(x_ans.data[1, :]))
tmp.data[1, :] = x_ans.data[1, :] - kappa*tmp.data[1, :]
t2 = perf_counter()
print(f'Quda dslash: {t2 - t1} sec')
