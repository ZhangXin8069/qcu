import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
from time import perf_counter

import cupy as cp

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))

from pyquda import init, core, quda, pyqcu, mpi
from pyquda.enum_quda import QudaParity
from pyquda.field import LatticeFermion
from pyquda.utils import gauge_utils
# from pyquda.pyqcu import dslashQuda_mpi

os.environ["QUDA_RESOURCE_PATH"] = ".cache"
latt_size = [32, 32, 32, 64]
# latt_size = [4,4,4,8]
# latt_size = [16, 16, 16, 32]
# latt_size = [2,2,2,4]

grid_size = [1, 1, 1, 2]
Lx, Ly, Lz, Lt = latt_size
Nd, Ns, Nc = 4, 4, 3
Gx, Gy, Gz, Gt = grid_size
latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
mpi.init(grid_size)

def dslash_my(b, buf, x_origion, dslash, QudaParity, kappa):
    
    quda.dslashQuda(buf.even_ptr, x_origion.odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
    quda.dslashQuda(b.odd_ptr, buf.even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
    
    b.data[1,:] = x_origion.data[1,:] - kappa*kappa*b.data[1,:]


def compare(round):
    # generate a vector p randomly
    
    # b = LatticeFermion(latt_size, cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128))
    # Mp = LatticeFermion(latt_size)
    # Mp1 = LatticeFermion(latt_size)
    # Mp2 = LatticeFermion(latt_size)

    x1 =  cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128)*1
    # x1 =  cp.ones((2,Lt, Lz, Ly, Lx//2, Ns, Nc),cp.complex128)
    # x1[1,0,0,0,0,0,0] = 1+1j;
    # x1[0,0,0,0,0,0,0] = 1+1j;
    x_origion = LatticeFermion(latt_size,x1)
    # x_origion.data[1,0,0,0,0,0,0] = 1+1j
    # x_origion.data[1,0,0,0,1,0,0] = 1+1j
    # print("x_origion = ", x_origion.data[0,0,0,0,0,0,:])

    r1 =  cp.zeros((2,Lt, Lz, Ly, Lx//2, Ns, Nc),cp.complex128)
    r = LatticeFermion(latt_size,r1)
    
    r2 =  cp.zeros((2,Lt, Lz, Ly, Lx//2, Ns, Nc),cp.complex128)
    r0 = LatticeFermion(latt_size,r2)
    
    t0 =  cp.zeros((2,Lt, Lz, Ly, Lx//2, Ns, Nc),cp.complex128)
    t = LatticeFermion(latt_size,t0)
    
    r_10 =  cp.zeros((2,Lt, Lz, Ly, Lx//2, Ns, Nc),cp.complex128)
    r_1 = LatticeFermion(latt_size,r_10)
    
    p0 =  cp.zeros((2,Lt, Lz, Ly, Lx//2, Ns, Nc),cp.complex128)
    p = LatticeFermion(latt_size,p0)
    
    buf0 =  cp.zeros((2,Lt, Lz, Ly, Lx//2, Ns, Nc),cp.complex128)
    buf = LatticeFermion(latt_size,buf0)
    
    buf10 =  cp.zeros((2,Lt, Lz, Ly, Lx//2, Ns, Nc),cp.complex128)
    buf1 = LatticeFermion(latt_size,buf10)
    
    Ap0 =  cp.zeros((2,Lt, Lz, Ly, Lx//2, Ns, Nc),cp.complex128)
    v = LatticeFermion(latt_size,Ap0)

    x0 =  cp.zeros((2,Lt, Lz, Ly, Lx//2, Ns, Nc),cp.complex128)
    x = LatticeFermion(latt_size,x0)
    
    s0 =  cp.zeros((2,Lt, Lz, Ly, Lx//2, Ns, Nc),cp.complex128)
    s = LatticeFermion(latt_size,s0)
    
    b0 =  cp.zeros((2,Lt, Lz, Ly, Lx//2, Ns, Nc),cp.complex128)
    # b0 =  cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128)*1
    # b0[1,0,0,0,0,1,0] = 1+1j;
    # b0[0,0,0,0,0,0,0] = 10;

    b = LatticeFermion(latt_size,b0)

 
    param = pyqcu.QcuParam()
    param.lattice_size = latt_size

    print('===============round ', round, '======================')

    # Set parameters in Dslash and use m=-3.5 to make kappa=1
    dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)
    kappa = 0.125
    # Generate gauge and then load it
    U = gauge_utils.gaussGauge(latt_size, 0)
    dslash.loadGauge(U)

    #生成Ax=b
    

    quda.dslashQuda(buf.even_ptr, x_origion.odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
    quda.dslashQuda(buf.odd_ptr, x_origion.even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
    
    b.data[:] = x_origion.data[:] - kappa*buf.data[:]
    # pyqcu.dslashQcu(buf.odd_ptr, b.even_ptr, U.data_ptr, param, 1)
    quda.dslashQuda(buf.odd_ptr, b.even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
    b.data[1,:] += kappa*buf.data[1,:]
    
    dslash_my(buf1, buf, x, dslash, QudaParity, kappa)
    r.data[1,:] = b.data[1,:] - buf1.data[1,:]
    print(r.data[1,1,1,1,1,1,1])
    r0.data[1,:] = r.data[1,:]
    alpha = 1.0
    w = 1.0
    rho = 1.0
    rho1 = 1.0


    # for i in range(1,800):
    #     # print((cp.inner(r.data[1,:].flatten().conjugate(),r.data[1,:].flatten())))
    #     rho = cp.inner(r0.data[1,:].flatten().conjugate(),r.data[1,:].flatten())
        
    #     beta = (rho/rho1)*(alpha/w)
    #     # print("beta = ",beta)
        
        
    #     p.data[1,:] = r.data[1,:] + beta*(p.data[1,:] - w*v.data[1,:])
        
    #     dslash_my(buf1, buf, p, dslash, QudaParity, kappa)
    #     v.data[1,:] = buf1.data[1,:]
        
    #     alpha = rho/(cp.inner(r0.data[1,:].flatten().conjugate(),v.data[1,:].flatten()))
    #     # print("alpha = ", alpha)
        
    #     s.data[1,:] = r.data[1,:] - alpha*v.data[1,:]
        
    #     dslash_my(buf1, buf, s, dslash, QudaParity, kappa)
    #     t.data[1,:] = buf1.data[1,:]
        
    #     w = (cp.inner(t.data[1,:].flatten().conjugate(),s.data[1,:].flatten()))/(cp.inner(t.data[1,:].flatten().conjugate(),t.data[1,:].flatten()))
        
    #     # print("s = ",(cp.inner(s.data[1,:].flatten().conjugate(),s.data[1,:].flatten())))
        
    #     # print("w = ", w)
    #     x.data[1,:] = x.data[1,:] + alpha * p.data[1,:] + w*s.data[1,:]
        
    #     # print("s = ",(cp.inner(s.data[1,:].flatten().conjugate(),s.data[1,:].flatten())))
        
    #     # print(cp.may_share_memory(s.data[1,:], x.data[1,:]))
    #     # print(cp.may_share_memory(s.data[1,:], p.data[1,:]))
        
    #     r.data[1,:] = s.data[1,:] - w*t.data[1,:]
    #     print((cp.inner(r.data[1,:].flatten().conjugate(),r.data[1,:].flatten())))
    #     rho1 = rho
        
    #     if (cp.inner(r.data[1,:].flatten().conjugate(),r.data[1,:].flatten()))<10e-6 :
    #         print(i)
    #         break
        
        
 
    
    pyqcu.dslashQcu(x.odd_ptr, b.odd_ptr, U.data_ptr, param, 0)
    
    
    
    
    print('difference: ', cp.linalg.norm(x.data[1,:] - x_origion.data[1,:]) / cp.linalg.norm(x.data[1,:]))
    #pyqcu.dslashQcu(buf.even_ptr, x_origion.odd_ptr, U.data_ptr, param, 0)
    #pyqcu.dslashQcu(buf.odd_ptr, x_origion.even_ptr, U.data_ptr, param, 1)
    # buf.data[1,:] = x_origion.data[1,:] - kappa*buf.data[1,:]
    # print("turns = ",turns,'\n')
    #print("x = \n",x.data[1,0,0,0,0,:],'\nx_origion = \n',x_origion.data[1,0,0,0,0,:])
        
    # t2 = perf_counter()
    # print(f'Quda dslash: {t2 - t1} sec')
    print(x.data[1,1,1,1,1,:])
    print(x_origion.data[1,1,1,1,1,:])




for i in range(0, 1 ):
    compare(i)