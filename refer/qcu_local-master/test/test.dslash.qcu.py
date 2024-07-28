from pyquda.utils import gauge_utils
from pyquda.field import LatticeFermion
from pyquda.enum_quda import QudaParity
from pyquda import init, core, quda, pyqcu as qcu
import os
import sys
from time import perf_counter

import cupy as cp
import numpy as np

test_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.insert(0, os.path.join(test_dir, ".."))


os.environ["QUDA_RESOURCE_PATH"] = ".cache"
init()

Lx, Ly, Lz, Lt = 32, 32, 32, 64
#Lx, Ly, Lz, Lt = 16,16,16,32
Nd, Ns, Nc = 4, 4, 3
latt_size = [Lx, Ly, Lz, Lt]

xi_0, nu = 1, 1
mass = -3.5
coeff_r, coeff_t = 1, 1


def compare(round):
    # generate a vector p randomly
    p = LatticeFermion(latt_size, cp.random.randn(
        Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128))

    Mp = LatticeFermion(latt_size)
    Mp1 = LatticeFermion(latt_size)

    print("===============round ", round, "======================")

    # Set parameters in Dslash and use m=-3.5 to make kappa=1
    #dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)
    dslash = core.getDslash(latt_size, mass, 1e-9, 1000, xi_0, nu, coeff_t,
                            coeff_r, multigrid=False, anti_periodic_t=False)  # anti_periodic_t=False
    # Generate gauge and then load it
    U = gauge_utils.gaussGauge(latt_size, round)
    dslash.loadGauge(U)

    cp.cuda.runtime.deviceSynchronize()
    t1 = perf_counter()
    quda.dslashQuda(Mp.even_ptr, p.odd_ptr, dslash.invert_param,
                    QudaParity.QUDA_EVEN_PARITY)
    quda.dslashQuda(Mp.odd_ptr, p.even_ptr, dslash.invert_param,
                    QudaParity.QUDA_ODD_PARITY)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    print(f"Quda dslash: {t2 - t1} sec")

    # then execute my code
    param = qcu.QcuParam()
    param.lattice_size = latt_size
    # U.data = cp.ascontiguousarray(U.data[:, :, :, :, :, :, :2, :])

    cp.cuda.runtime.deviceSynchronize()
    t1 = perf_counter()
    qcu.dslashQcu(Mp1.even_ptr, p.odd_ptr, U.data_ptr, param, 0)
    #qcu.dslashQcu(Mp1.even_ptr, p.even_ptr, U.data_ptr, param, 0)
    qcu.dslashQcu(Mp1.odd_ptr, p.even_ptr, U.data_ptr, param, 1)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    print(f"QCU dslash: {t2 - t1} sec")

    print("difference: ", cp.linalg.norm(
        Mp1.data - Mp.data) / cp.linalg.norm(Mp.data))
    #print("Mp.norm: ", cp.linalg.norm(Mp.data))
    #print("Mp1.norm: ", cp.linalg.norm(Mp1.data))
    #out = Mp.lexico()
    #out1 = Mp1.lexico()
    #print('Mp[63,0,0,0]: \n', out[63,0,0,0])
    #print('Mp1[63,0,0,0]: \n', out1[63,0,0,0])
    '''for x in range(0, Lx):
        for y in range(0, Ly):
            for z in range(0, Lz):
                for t in range(0, Lt):
                    if cp.linalg.norm(out[t,z,y,x] - out1[t,z,y,x])>0.000001 :
                        print(f'tuple {x}, {y},{z},{t}')
                        print('Mp:')
                        print(out[t,z,y,x])
                        print('Mp1:')
                        print(out1[t,z,y,x])'''

#    print('Mp[63,0,0,0]: \n', out[63,0,0,0])
#    print('Mp1[63,0,0,0]: \n', out1[63,0,0,0])
    return U.lexico()


def gamma(num):
    import numpy as np
    gamma1 = np.array([[0, 0, 0, 1j], [0, 0, 1j, 0], [
                      0, -1j, 0, 0], [-1j, 0, 0, 0]], dtype=np.complex128)
    gamma2 = np.array([[0, 0, 0, -1], [0, 0, 1, 0],
                      [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=np.complex128)
    gamma3 = np.array([[0, 0, 1j, 0], [0, 0, 0, -1j],
                      [-1j, 0, 0, 0], [0, 1j, 0, 0]], dtype=np.complex128)
    gamma4 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [
                      0, 1, 0, 0]], dtype=np.complex128)
    if num == 1:
        return gamma1
    elif num == 2:
        return gamma2
    elif num == 3:
        return gamma3
    else:
        return gamma4


def move(t, z, y, x, pos):
    pos_x, pos_y, pos_z, pos_t = x, y, z, t
    if pos == 1:
        pos_x = (x+1) % Lx
    elif pos == -1:
        pos_x = (x+Lx-1) % Lx
    elif pos == 2:
        # print(f'y={(y+1)%Ly}')
        pos_y = (y+1) % Ly
        # print(f'pos_y={pos_y}')
    elif pos == -2:
        pos_y = (y+Ly-1) % Ly
    elif pos == 3:
        pos_z = (z+1) % Lz
    elif pos == -3:
        pos_z = (z+Lz-1) % Lz
    elif pos == 4:
        pos_t = (t+1) % Lt
    else:
        pos_t = (t+Lt-1) % Lt
    return pos_t, pos_z, pos_y, pos_x


def calcF(U, t, z, y, x, mu, nu):
    pos_t1, pos_z1, pos_y1, pos_x1 = move(t, z, y, x, mu)
    pos_t2, pos_z2, pos_y2, pos_x2 = move(t, z, y, x, nu)
    temp = U[mu-1, t, z, y, x]@U[nu-1, pos_t1, pos_z1, pos_y1, pos_x1]@U[mu-1,
                                                                         pos_t2, pos_z2, pos_y2, pos_x2].T.conjugate()@U[nu-1, t, z, y, x].T.conjugate()

    pos_t2, pos_z2, pos_y2, pos_x2 = move(t, z, y, x, -mu)
    pos_t1, pos_z1, pos_y1, pos_x1 = move(pos_t2, pos_z2, pos_y2, pos_x2, nu)
    temp += U[nu-1, t, z, y, x]@U[mu-1, pos_t1, pos_z1, pos_y1, pos_x1].T.conjugate()@U[nu-1,
                                                                                        pos_t2, pos_z2, pos_y2, pos_x2].T.conjugate()@U[mu-1, pos_t2, pos_z2, pos_y2, pos_x2]

    pos_t1, pos_z1, pos_y1, pos_x1 = move(t, z, y, x, -mu)
    pos_t2, pos_z2, pos_y2, pos_x2 = move(pos_t1, pos_z1, pos_y1, pos_x1, -nu)
    pos_t3, pos_z3, pos_y3, pos_x3 = move(t, z, y, x, -nu)
    temp += U[mu-1, pos_t1, pos_z1, pos_y1, pos_x1].T.conjugate()@U[nu-1, pos_t2, pos_z2, pos_y2,
                                                                    pos_x2].T.conjugate()@U[mu-1, pos_t2, pos_z2, pos_y2, pos_x2]@U[nu-1, pos_t3, pos_z3, pos_y3, pos_x3]

    pos_t1, pos_z1, pos_y1, pos_x1 = move(t, z, y, x, -nu)
    pos_t2, pos_z2, pos_y2, pos_x2 = move(pos_t1, pos_z1, pos_y1, pos_x1, mu)
    temp += U[nu-1, pos_t1, pos_z1, pos_y1, pos_x1].T.conjugate()@U[mu-1, pos_t1, pos_z1,
                                                                    pos_y1, pos_x1]@U[nu-1, pos_t2, pos_z2, pos_y2, pos_x2]@U[mu-1, t, z, y, x].T.conjugate()
    result = temp - temp.T.conjugate()
    return result


def calcSigma(mu, nu):
    sigma12 = gamma(1)@gamma(2) - gamma(2)@gamma(1)
    sigma13 = gamma(1)@gamma(3) - gamma(3)@gamma(1)
    sigma14 = gamma(1)@gamma(4) - gamma(4)@gamma(1)
    sigma23 = gamma(2)@gamma(3) - gamma(3)@gamma(2)
    sigma24 = gamma(2)@gamma(4) - gamma(4)@gamma(2)
    sigma34 = gamma(3)@gamma(4) - gamma(4)@gamma(3)
    if mu == 1 and nu == 2:
        return sigma12
    elif mu == 1 and nu == 3:
        return sigma13
    elif mu == 1 and nu == 4:
        return sigma14
    elif mu == 2 and nu == 3:
        return sigma23
    elif mu == 2 and nu == 4:
        return sigma24
    elif mu == 3 and nu == 4:
        return sigma34
    else:
        return gamma(1)@gamma(1) - gamma(1)@gamma(1)


def calculateClover(U):
    import numpy as np
    t, z, y, x = 31, 0, 0, 0
    result = np.zeros((Ns*Nc, Ns*Nc), dtype=np.complex128)
    for i in range(1, 5):
        for j in range(i+1, 5):
            temp = np.tensordot(calcSigma(i, j), calcF(
                U, t, z, y, x, i, j), axes=0)
            result += 2 * (temp.swapaxes(1, 2).reshape(12, 12))
    #print('clover[0,0,0,1] = >>>>>>>>>>>')
    # print(result)
    # print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    return result


for i in range(0, 5):
    U = compare(i)
    #clover = calculateClover(U)
    #clover = calculateClover(U)/(-32)
    # for ii in range(0, 12):
    #    clover[ii][ii] += 1
    #a = np.linalg.inv(clover)
    #src = np.array([[0.977128 + 0.047261j], [0.139568 + 0.132403j], [-0.031455 + 0.070626j], [0.000000 + 0.000000j],[0.000000 + 0.000000j], [0.000000 + 0.000000j], [-0.977128 + -0.047261j], [-0.139568 + -0.132403j], [0.031455 + -0.070626j], [0.000000 + 0.000000j], [0.000000 + 0.000000j], [0.000000 + 0.000000j]])
    # print(a)
    # print(a@src)
