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
a = 1
xi_0, nu = 1, 1
coeff_r, coeff_t = 1, 1
mass = -3.5
# kappa = 1 / (2*mass+8)

gamma0 = [
    [
        [0,0,0,1j],
        [0,0,1j,0],
        [0,-1j,0,0],
        [-1j,0,0,0]
    ],
    [
        [0,0,0,-1],
        [0,0,1,0],
        [0,1,0,0],
        [-1,0,0,0]
    ],
    [
        [0,0,1j,0],
        [0,0,0,-1j],
        [-1j,0,0,0],
        [0,1j,0,0]
    ],
    [
        [0,0,1,0],
        [0,0,0,1],
        [1,0,0,0],
        [0,1,0,0]
    ]
]

gamma = cp.array(gamma0)

def compare(round):
    # generate a vector p randomly
    p = LatticeFermion(latt_size, cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128))
    # p0 =  cp.zeros((2,Lt, Lz, Ly, Lx//2, Ns, Nc),cp.complex128)
    # p0[1,1,0,0,0,0,0] = 1+0.4j
    # p0[1,-1,0,0,0,0,0] = 1
    # p0[0,0,0,0,0,:] = cp.random.randn(Ns, Nc * 2).view(cp.complex128)
    # p = LatticeFermion(latt_size,p0)
    Mp = LatticeFermion(latt_size)
    Mp_p = LatticeFermion(latt_size)
    Mp1 = LatticeFermion(latt_size)

    print('===============round ', round, '======================')
    print("######p[0,0,0,1]:\n",p.lexico()[0,0,0,1])

    # Set parameters in Dslash and use m=-3.5 to make kappa=1

    dslash1 = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)
    dslash = core.getDslash(latt_size, mass, 1e-9, 20, xi_0, nu, coeff_t, coeff_r, multigrid=False)
    dslash = core.getDslash(latt_size, -3.5, 1e-9, 200, 1,1)
    dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)
    # Generate gauge and then load it
    print(1)
    U = gauge_utils.gaussGauge(latt_size,0)
    print(2)

    # # Set parameters in Dslash and use m=-3.5 to make kappa=1
    # dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)
    # kappa = 0.125
    # # Generate gauge and then load it
    # U = gauge_utils.gaussGauge(latt_size, 0)
    # dslash.loadGauge(U)
    print(4)
    dslash.loadGauge(U)
    dslash1.loadGauge(U)
    print(3)

    param = pyqcu.QcuParam()
    param.lattice_size = latt_size
    
    cp.cuda.runtime.deviceSynchronize()
    t1 = perf_counter()
    quda.dslashQuda(Mp.even_ptr, p.odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
    quda.dslashQuda(Mp.odd_ptr, p.even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
    
    quda.dslashQuda(Mp_p.even_ptr, p.odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
    quda.dslashQuda(Mp_p.odd_ptr, p.even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    print(f'Quda dslash: {t2 - t1} sec')

    print("Mp= \n",Mp.data[0,0,0,0,0,:])
    # then execute my code
    param = pyqcu.QcuParam()
    param.lattice_size = latt_size

    cp.cuda.runtime.deviceSynchronize()
    t1 = perf_counter()
    pyqcu.dslashCloverQcu(Mp1.even_ptr, p.odd_ptr, U.data_ptr, param, 0)
    pyqcu.dslashCloverQcu(Mp1.odd_ptr, p.even_ptr, U.data_ptr, param, 1)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    # x=0,0,0,0
    # u,v = 3,2
    buf = U.data[3,0,0,0,0,0,:]@U.data[2,1,1,0,0,0,:]@U.data[3,1,0,1,0,0,:].T.conjugate()@U.data[2,0,0,0,0,0,:].T.conjugate()
    buf += U.data[2,0,0,0,0,0,:]@U.data[3,0,-1,1,0,0,:].T.conjugate()@U.data[2,1,-1,0,0,0,:].T.conjugate()@U.data[3,1,-1,0,0,0,:]
    buf += U.data[3,1,-1,0,0,0,:].T.conjugate()@U.data[2,0,-1,-1,0,0,:].T.conjugate()@U.data[3,0,-1,-1,0,0,:]@U.data[2,1,0,-1,0,0,:]
    buf += U.data[2,1,0,-1,0,0,:].T.conjugate()@U.data[3,1,0,-1,0,0,:]@U.data[2,0,1,-1,0,0,:]@U.data[3,0,0,0,0,0,:].T.conjugate()
    buf -= buf.T.conjugate()

    T = cp.kron(gamma[3,:]@gamma[2,:],buf)

    # u,v = 3,1
    buf = U.data[3,0,0,0,0,0,:]@U.data[1,1,1,0,0,0,:]@U.data[3,1,0,0,1,0,:].T.conjugate()@U.data[1,0,0,0,0,0,:].T.conjugate()
    buf += U.data[1,0,0,0,0,0,:]@U.data[3,0,-1,0,1,0,:].T.conjugate()@U.data[1,1,-1,0,0,0,:].T.conjugate()@U.data[3,1,-1,0,0,0,:]
    buf += U.data[3,1,-1,0,0,0,:].T.conjugate()@U.data[1,0,-1,0,-1,0,:].T.conjugate()@U.data[3,0,-1,0,-1,0,:]@U.data[1,1,0,0,-1,0,:]
    buf += U.data[1,1,0,0,-1,0,:].T.conjugate()@U.data[3,1,0,0,-1,0,:]@U.data[1,0,1,0,-1,0,:]@U.data[3,0,0,0,0,0,:].T.conjugate()
    buf -= buf.T.conjugate()
    T += cp.kron(gamma[3,:]@gamma[1,:],buf)



    # u,v = 3,0
    buf = U.data[3,0,0,0,0,0,:]@U.data[0,1,1,0,0,0,:]@U.data[3,1,0,0,0,0,:].T.conjugate()@U.data[0,0,0,0,0,0,:].T.conjugate()
    buf += U.data[0,0,0,0,0,0,:]@U.data[3,0,-1,0,0,0,:].T.conjugate()@U.data[0,1,-1,0,0,0,:].T.conjugate()@U.data[3,1,-1,0,0,0,:]
    buf += U.data[3,1,-1,0,0,0,:].T.conjugate()@U.data[0,0,-1,0,0,-1,:].T.conjugate()@U.data[3,0,-1,0,0,-1,:]@U.data[0,1,0,0,0,-1,:]
    buf += U.data[0,1,0,0,0,-1,:].T.conjugate()@U.data[3,1,0,0,0,-1,:]@U.data[0,0,1,0,0,-1,:]@U.data[3,0,0,0,0,0,:].T.conjugate()
    buf -= buf.T.conjugate()
    T += cp.kron(gamma[3,:]@gamma[0,:],buf)
 
    # u,v = 2,1
    buf = U.data[2,0,0,0,0,0,:]@U.data[1,1,0,1,0,0,:]@U.data[2,1,0,0,1,0,:].T.conjugate()@U.data[1,0,0,0,0,0,:].T.conjugate()
    buf += U.data[1,0,0,0,0,0,:]@U.data[2,0,0,-1,1,0,:].T.conjugate()@U.data[1,1,0,-1,0,0,:].T.conjugate()@U.data[2,1,0,-1,0,0,:]
    buf += U.data[2,1,0,-1,0,0,:].T.conjugate()@U.data[1,0,0,-1,-1,0,:].T.conjugate()@U.data[2,0,0,-1,-1,0,:]@U.data[1,1,0,0,-1,0,:]
    buf += U.data[1,1,0,0,-1,0,:].T.conjugate()@U.data[2,1,0,0,-1,0,:]@U.data[1,0,0,1,-1,0,:]@U.data[2,0,0,0,0,0,:].T.conjugate()
    buf -= buf.T.conjugate()
    T += cp.kron(gamma[2,:]@gamma[1,:],buf)

    # u,v = 2,0
    buf = U.data[2,0,0,0,0,0,:]@U.data[0,1,0,1,0,0,:]@U.data[2,1,0,0,0,0,:].T.conjugate()@U.data[0,0,0,0,0,0,:].T.conjugate()
    buf += U.data[0,0,0,0,0,0,:]@U.data[2,0,0,-1,0,0,:].T.conjugate()@U.data[0,1,0,-1,0,0,:].T.conjugate()@U.data[2,1,0,-1,0,0,:]
    buf += U.data[2,1,0,-1,0,0,:].T.conjugate()@U.data[0,0,0,-1,0,-1,:].T.conjugate()@U.data[2,0,0,-1,0,-1,:]@U.data[0,1,0,0,0,-1,:]
    buf += U.data[0,1,0,0,0,-1,:].T.conjugate()@U.data[2,1,0,0,0,-1,:]@U.data[0,0,0,1,0,-1,:]@U.data[2,0,0,0,0,0,:].T.conjugate()
    buf -= buf.T.conjugate()
    T += cp.kron(gamma[2,:]@gamma[0,:],buf)

    
    # u,v = 1,0
    buf = U.data[1,0,0,0,0,0,:]@U.data[0,1,0,0,1,0,:]@U.data[1,1,0,0,0,0,:].T.conjugate()@U.data[0,0,0,0,0,0,:].T.conjugate()
    buf += U.data[0,0,0,0,0,0,:]@U.data[1,0,0,0,-1,0,:].T.conjugate()@U.data[0,1,0,0,-1,0,:].T.conjugate()@U.data[1,1,0,0,-1,0,:]
    buf += U.data[1,1,0,0,-1,0,:].T.conjugate()@U.data[0,0,0,0,-1,-1,:].T.conjugate()@U.data[1,0,0,0,-1,-1,:]@U.data[0,1,0,0,0,-1,:]
    buf += U.data[0,1,0,0,0,-1,:].T.conjugate()@U.data[1,1,0,0,0,-1,:]@U.data[0,0,0,0,1,-1,:]@U.data[1,0,0,0,0,0,:].T.conjugate()
    buf -= buf.T.conjugate()

    T += cp.kron(gamma[1,:]@gamma[0,:],buf)


    A = cp.linalg.inv(cp.eye(12,12) - T/8)
    # print("I-T/8:",(cp.eye(12,12) - T/8)[0:3,0:6])
    # print("-A:",A[6,6:12])
    # print(U.data[3,0,0,0,0,0,:])
    # print("######Mp[0,0,0,0]:\n",Mp_p.lexico()[0,0,0,0])
    print("######Mp1[0,0,0,0]:\n",Mp1.lexico()[0,0,0,0])
    print("######Mp2[0,0,0,0]:\n",(A@(Mp_p.data[0,0,0,0,0,:].reshape(12,1))).reshape(4,3))
    
    print(f'QCU dslash: {t2 - t1} sec')
    print('difference: ', cp.linalg.norm(Mp1.data - Mp.data) / cp.linalg.norm(Mp.data))
    # print('difference: ', cp.linalg.norm(Mp1.data - p.data) / cp.linalg.norm(Mp.data))
    # for i in range(4):
    #     for j in range(i):
    #         if True:
    #             print(i,"     ",j,":\n")
    #             print((gamma[i,:]@gamma[j,:])[0:5,0:5])






# for i in range(0, 5):
compare(0)