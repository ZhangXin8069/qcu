#include "qcu_clover_dslash.cuh"
#include "qcu_point.cuh"
#include "qcu_complex.cuh"
#include "qcu_communicator.cuh"
#include "qcu.h"
#include <chrono>
// #define DEBUG

extern MPICommunicator *mpi_comm;

static __device__ inline void reconstructSU3(Complex *su3)
{
  su3[6] = (su3[1] * su3[5] - su3[2] * su3[4]).conj();
  su3[7] = (su3[2] * su3[3] - su3[0] * su3[5]).conj();
  su3[8] = (su3[0] * su3[4] - su3[1] * su3[3]).conj();
}

// FROM newbing
// 定义一个函数，用于交换两行
static __device__ void swapRows(Complex* matrix, int row1, int row2) {
  Complex temp;
  for (int i = 0; i < N * 2; i++) {
    temp = matrix[row1 * N * 2 + i];
    matrix[row1 * N * 2 + i] = matrix[row2 * N * 2 + i];
    matrix[row2 * N * 2 + i] = temp;
  }
}
// 定义一个函数，用于将一行除以一个数
static __device__ void divideRow(Complex* matrix, int row, Complex num) {
  for (int i = 0; i < N * 2; i++) {
    matrix[row * N * 2 + i] /= num;
  }
}
// 定义一个函数，用于将一行减去另一行乘以一个数
static __device__ void subtractRow(Complex* matrix, int row1, int row2, Complex num) {
  for (int i = 0; i < N * 2; i++) {
    matrix[row1 * N * 2 + i] -= num * matrix[row2 * N * 2 + i];
  }
}

// 定义一个函数，用于求逆矩阵
static __device__ void inverseMatrix(Complex* matrix, Complex* result) {
  // 创建一个单位矩阵
  // double* identity = (double*)malloc(sizeof(double) * N * N);
  Complex identity[N*N];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (i == j) {
        identity[i * N + j] = Complex(1, 0);
      } else {
        identity[i * N + j].clear2Zero();
      }
    }
  }
  // 将原矩阵和单位矩阵平接在一起，形成增广矩阵
  Complex augmented[N*N*2];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N * 2; j++) {
      if (j < N) {
        augmented[i * N * 2 + j] = matrix[i * N + j];
      } else {
        augmented[i * N * 2 + j] = identity[i * N + (j - N)];
      }
    }
  }

  // 对增广矩阵进行高斯消元法
  for (int i = 0; i < N; i++) {
    // 如果对角线上的元素为0，就交换两行
    if (augmented[i * N * 2 + i] == Complex(0, 0)) {
      for (int j = i + 1; j < N; j++) {
        if (augmented[j * N * 2 + i] != Complex(0,0)) {
          swapRows(augmented, i, j);
          break;
        }
      }
    }

    // 如果对角线上的元素不为1，就将该行除以该元素
    if (augmented[i * N * 2 + i] != Complex(1,0)) {
      divideRow(augmented, i, augmented[i * N * 2 + i]);
    }

    // 将其他行减去该行乘以相应的系数，使得该列除了对角线上的元素外都为0
    for (int j = 0; j < N; j++) {
      if (j != i) {
        subtractRow(augmented, j, i, augmented[j * N * 2 + i]);
      }
    }
  }

  // 从增广矩阵中分离出逆矩阵
  // double* inverse = (double*)malloc(sizeof(double) * N * N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      // inverse[i * N + j] = augmented[i * N * 2 + (j + N)];
      result[i * N + j] = augmented[i * N * 2 + (j + N)];
    }
  }
}

static __device__ inline void loadGauge(Complex* u_local, void* gauge_ptr, int direction, const Point& p, int Lx, int Ly, int Lz, int Lt) {
  Complex* u = p.getPointGauge(static_cast<Complex*>(gauge_ptr), direction, Lx, Ly, Lz, Lt);
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    u_local[i] = u[i];
  }
  reconstructSU3(u_local);
}
static __device__ inline void loadVector(Complex* src_local, void* fermion_in, const Point& p, int Lx, int Ly, int Lz, int Lt) {
  Complex* src = p.getPointVector(static_cast<Complex *>(fermion_in), Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src[i];
  }
}




static __device__ void copyGauge (Complex* dst, Complex* src) {
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    dst[i] = src[i];
  }
  reconstructSU3(dst);
}


// WARP version, no sync
static __device__ void storeVectorBySharedMemory(void* shared_ptr, Complex* origin, Complex* result) {
  // result is register variable
  // __shared__ double shared_buffer[2 * Ns * Nc * BLOCK_SIZE];
    // __shared__ double shared_buffer[2 * Ns * Nc * BLOCK_SIZE];
  double* shared_buffer = static_cast<double*>(shared_ptr);
  int thread = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_index = (thread - thread / BLOCK_SIZE * BLOCK_SIZE) / WARP_SIZE;//thread % BLOCK_SIZE / WARP_SIZE;
  Complex* shared_dst = reinterpret_cast<Complex*>(shared_buffer) + threadIdx.x * Ns * Nc;
  Complex* warp_dst = origin + (thread / WARP_SIZE * WARP_SIZE) * Ns * Nc;
  double* double_dst = reinterpret_cast<double*>(warp_dst);

  // store result to shared memory
  for (int i = 0; i < Ns * Nc; i++) {
    shared_dst[i] = result[i];
  }
  // store result of shared memory to global memory
  for (int i = threadIdx.x - threadIdx.x / WARP_SIZE * WARP_SIZE; i < WARP_SIZE * Ns * Nc * 2; i += WARP_SIZE) {
    double_dst[i] = shared_buffer[warp_index * WARP_SIZE * Ns * Nc * 2 + i];
  }
}



// WARP version, no sync
static __device__ void loadVectorBySharedMemory(void* shared_ptr, Complex* origin, Complex* result) {
  // result is register variable
  // __shared__ double shared_buffer[2 * Ns * Nc * BLOCK_SIZE];
  double* shared_buffer = static_cast<double*>(shared_ptr);
  int thread = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_index = (thread - thread / BLOCK_SIZE * BLOCK_SIZE) / WARP_SIZE;//thread % BLOCK_SIZE / WARP_SIZE;
  Complex* shared_dst = reinterpret_cast<Complex*>(shared_buffer) + threadIdx.x * Ns * Nc;
  Complex* warp_dst = origin + (thread / WARP_SIZE * WARP_SIZE) * Ns * Nc;
  double* double_dst = reinterpret_cast<double*>(warp_dst);

  // store result of shared memory to global memory
  for (int i = threadIdx.x - threadIdx.x / WARP_SIZE * WARP_SIZE; i < WARP_SIZE * Ns * Nc * 2; i += WARP_SIZE) {
    shared_buffer[warp_index * WARP_SIZE * Ns * Nc * 2 + i] = double_dst[i];
  }
  // store result to shared memory
  for (int i = 0; i < Ns * Nc; i++) {
    result[i] = shared_dst[i];
  }
}


__device__ __host__ void gaugeMul (Complex* result, Complex* u1, Complex* u2) {
  Complex temp;
  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      temp.clear2Zero();
      for (int k = 0; k < Nc; k++) {
        // result[i * Nc + j] = u1[i * Nc + k] * u2[k * Nc + j];
        temp += u1[i * Nc + k] * u2[k * Nc + j];
      }
      result[i * Nc + j] = temp;
    }
  }
}
__device__ __host__ inline void gaugeAddAssign (Complex* u1, Complex* u2) {
  u1[0] += u2[0];  u1[1] += u2[1];  u1[2] += u2[2];
  u1[3] += u2[3];  u1[4] += u2[4];  u1[5] += u2[5];
  u1[6] += u2[6];  u1[7] += u2[7];  u1[8] += u2[8];
}
__device__ __host__ inline void gaugeSubAssign (Complex* u1, Complex* u2) {
  u1[0] -= u2[0];  u1[1] -= u2[1];  u1[2] -= u2[2];
  u1[3] -= u2[3];  u1[4] -= u2[4];  u1[5] -= u2[5];
  u1[6] -= u2[6];  u1[7] -= u2[7];  u1[8] -= u2[8];
}
__device__ __host__ inline void gaugeAssign(Complex* u1, Complex* u2) {
  u1[0] = u2[0];  u1[1] = u2[1];  u1[2] = u2[2];
  u1[3] = u2[3];  u1[4] = u2[4];  u1[5] = u2[5];
  u1[6] = u2[6];  u1[7] = u2[7];  u1[8] = u2[8];
}
__device__ __host__ inline void gaugeTransposeConj(Complex* u) {
  Complex temp;
  u[0] = u[0].conj(); u[4] = u[4].conj(); u[8] = u[8].conj();  // diag
  temp = u[1];  u[1] = u[3].conj(); u[3] = temp.conj();
  temp = u[2];  u[2] = u[6].conj(); u[6] = temp.conj();
  temp = u[5];  u[5] = u[7].conj(); u[7] = temp.conj();
}

// assume tensor_product is[Ns * Nc * Ns * Nc] and gauge is [Nc * Nc]
__device__ __host__ void tensorProductAddition(Complex* tensor_product, Complex* gauge, int mu, int nu, Complex co_eff = Complex(1, 0)) {
  // 12 times 12 matrix -----> 12 * 6 matrix
  if (mu==0 && nu==1) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        tensor_product[               i*Ns*Nc/2 +      j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
        tensor_product[  Nc*Ns*Nc/2 + i*Ns*Nc/2 + Nc + j] += co_eff * Complex(0,  1) * gauge[i*Nc + j] * 2;
        tensor_product[2*Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
        tensor_product[3*Nc*Ns*Nc/2 + i*Ns*Nc/2 + Nc + j] += co_eff * Complex(0,  1) * gauge[i*Nc + j] * 2;
      }
    }
  } else if (mu==0 && nu==2) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        tensor_product[               i*Ns*Nc/2 + Nc + j] += co_eff * Complex(-1, 0) * gauge[i*Nc + j] * 2;
        tensor_product[  Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex( 1, 0) * gauge[i*Nc + j] * 2;
        tensor_product[2*Nc*Ns*Nc/2 + i*Ns*Nc/2 + Nc + j] += co_eff * Complex(-1, 0) * gauge[i*Nc + j] * 2;
        tensor_product[3*Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex( 1, 0) * gauge[i*Nc + j] * 2;
      }
    }
  } else if (mu==0 && nu==3) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        tensor_product[               i*Ns*Nc/2 + Nc + j] += co_eff * Complex(0,  1) * gauge[i*Nc + j] * 2;
        tensor_product[  Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex(0,  1) * gauge[i*Nc + j] * 2;
        tensor_product[2*Nc*Ns*Nc/2 + i*Ns*Nc/2 + Nc + j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
        tensor_product[3*Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
      }
    }
  } else if (mu==1 && nu==2) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        tensor_product[               i*Ns*Nc/2 + Nc + j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
        tensor_product[  Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
        tensor_product[2*Nc*Ns*Nc/2 + i*Ns*Nc/2 + Nc + j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
        tensor_product[3*Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
      }
    }
  } else if (mu==1 && nu==3) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        tensor_product[               i*Ns*Nc/2 + Nc + j] += co_eff * Complex(-1, 0) * gauge[i*Nc + j] * 2;
        tensor_product[  Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex( 1, 0) * gauge[i*Nc + j] * 2;
        tensor_product[2*Nc*Ns*Nc/2 + i*Ns*Nc/2 + Nc + j] += co_eff * Complex( 1, 0) * gauge[i*Nc + j] * 2;
        tensor_product[3*Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex(-1, 0) * gauge[i*Nc + j] * 2;
      }
    }
  } else if (mu==2 && nu==3) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        tensor_product[               i*Ns*Nc/2 +      j] += co_eff * Complex(0,  1) * gauge[i*Nc + j] * 2;
        tensor_product[  Nc*Ns*Nc/2 + i*Ns*Nc/2 + Nc + j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
        tensor_product[2*Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
        tensor_product[3*Nc*Ns*Nc/2 + i*Ns*Nc/2 + Nc + j] += co_eff * Complex(0,  1) * gauge[i*Nc + j] * 2;
      }
    }
  }
}


class Clover {
private:
  Complex* first_item;
  Complex* second_item;
  Complex* third_item;
  Complex* fourth_item;
  Complex* dst_;         // addr to store

  Complex* origin_gauge;
  Complex* shift_gauge[Nd][2];
  Complex* shift_shift_gauge[6][4];
public:
  __device__ int indexTwiceShift(int mu, int nu) {

    assert(mu >= X_DIRECTION && mu <= T_DIRECTION);
    assert(nu >= X_DIRECTION && nu <= T_DIRECTION);
    assert(mu < nu);
    int index = -1;
    switch (mu) {
      case X_DIRECTION:
        if (nu == Y_DIRECTION) index = 0;
        else if (nu == Z_DIRECTION) index = 1;
        else index = 2;
        break;
      case Y_DIRECTION:
        if (nu == Z_DIRECTION) index = 3;
        else index = 4;
        break;
      case Z_DIRECTION:
        index = 5;
        break;
      default:
        break;
    }
    return index;
  }
  __device__ void initialize(Complex* p_origin_gauge, Complex** p_shift_gauge, Complex** p_shift_shift_gauge) {
    origin_gauge = p_origin_gauge;
    for (int i = 0; i < Nd; i++) {
      for (int j = 0; j < 2; j++) {
        shift_gauge[i][j] = p_shift_gauge[i * 2 + j];// mpi_comm->gauge_shift[i][j];
      }
    }
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 4; j++) {
        shift_shift_gauge[i][j] = p_shift_shift_gauge[i * 4 + j];//mpi_comm->gauge_twice_shift[i][j];
      }
    }
  }
  __device__ Clover(Complex* p_origin_gauge, Complex** p_shift_gauge, Complex** p_shift_shift_gauge) {
    initialize(p_origin_gauge, p_shift_gauge, p_shift_shift_gauge);
  }
  __device__ void setDst(Complex* dst) {
    dst_ = dst;
  }
  __device__ Complex* getDst() const{
    return dst_;
  }

  __device__ void F_1(Complex* partial_result, const Point &point, int mu, int nu, int Lx, int Ly, int Lz, int Lt) {
    Complex lhs[Nc * Nc];
    Complex rhs[Nc * Nc];
    first_item = point.getPointGauge(origin_gauge, mu, Lx, Ly, Lz, Lt);
    second_item = point.getPointGauge(shift_gauge[mu][FRONT], nu, Lx, Ly, Lz, Lt);
    third_item = point.getPointGauge(shift_gauge[nu][FRONT], mu, Lx, Ly, Lz, Lt);
    fourth_item = point.getPointGauge(origin_gauge, nu, Lx, Ly, Lz, Lt);
    copyGauge (lhs, first_item);
    copyGauge (rhs, second_item);
    gaugeMul(partial_result, lhs, rhs);
    gaugeAssign(lhs, partial_result);
    copyGauge (rhs, third_item);
    gaugeTransposeConj(rhs);

    gaugeMul(partial_result, lhs, rhs);
    gaugeAssign(lhs, partial_result);
    copyGauge (rhs, fourth_item);
    gaugeTransposeConj(rhs);
    gaugeMul(partial_result, lhs, rhs);
  }
  __device__ void F_2(Complex* partial_result, const Point &point, int mu, int nu, int Lx, int Ly, int Lz, int Lt) {
    Complex lhs[Nc * Nc];
    Complex rhs[Nc * Nc];
    first_item = point.getPointGauge(shift_gauge[mu][BACK], mu, Lx, Ly, Lz, Lt);
    second_item = point.getPointGauge(origin_gauge, nu, Lx, Ly, Lz, Lt);
    third_item = point.getPointGauge(shift_shift_gauge[indexTwiceShift(mu, nu)][2], mu, Lx, Ly, Lz, Lt);
    fourth_item = point.getPointGauge(shift_gauge[mu][BACK], nu, Lx, Ly, Lz, Lt);
    copyGauge (lhs, second_item);
    copyGauge (rhs, third_item);

    gaugeTransposeConj(rhs);

    gaugeMul(partial_result, lhs, rhs);

    gaugeAssign(lhs, partial_result);
    copyGauge (rhs, fourth_item);
    gaugeTransposeConj(rhs);

    gaugeMul(partial_result, lhs, rhs);

    gaugeAssign(lhs, partial_result);
    copyGauge (rhs, first_item);
    gaugeMul(partial_result, lhs, rhs);
  }
  __device__ void F_3(Complex* partial_result, const Point &point, int mu, int nu, int Lx, int Ly, int Lz, int Lt) {
    Complex lhs[Nc * Nc];
    Complex rhs[Nc * Nc];
    first_item = point.getPointGauge(shift_shift_gauge[indexTwiceShift(mu, nu)][3], mu, Lx, Ly, Lz, Lt);
    second_item = point.getPointGauge(shift_gauge[nu][BACK], nu, Lx, Ly, Lz, Lt);
    third_item = point.getPointGauge(shift_gauge[mu][BACK], mu, Lx, Ly, Lz, Lt);
    fourth_item = point.getPointGauge(shift_shift_gauge[indexTwiceShift(mu, nu)][3], nu, Lx, Ly, Lz, Lt);
    copyGauge (lhs, third_item);
    gaugeTransposeConj(lhs);
    copyGauge (rhs, fourth_item);
    gaugeTransposeConj(rhs);

    gaugeMul(partial_result, lhs, rhs);
    gaugeAssign(lhs, partial_result);
    copyGauge (rhs, first_item);
    gaugeMul(partial_result, lhs, rhs);
    gaugeAssign(lhs, partial_result);
    copyGauge (rhs, second_item);
    gaugeMul(partial_result, lhs, rhs);
  }
  __device__ void F_4(Complex* partial_result, const Point &point, int mu, int nu, int Lx, int Ly, int Lz, int Lt) {
    Complex lhs[Nc * Nc];
    Complex rhs[Nc * Nc];
    first_item = point.getPointGauge(shift_gauge[nu][BACK], mu, Lx, Ly, Lz, Lt);
    second_item = point.getPointGauge(shift_shift_gauge[indexTwiceShift(mu, nu)][1], nu, Lx, Ly, Lz, Lt);
    third_item = point.getPointGauge(origin_gauge, mu, Lx, Ly, Lz, Lt);
    fourth_item = point.getPointGauge(shift_gauge[nu][BACK], nu, Lx, Ly, Lz, Lt);
    copyGauge (lhs, fourth_item);
    gaugeTransposeConj(lhs);
    copyGauge (rhs, first_item);
    gaugeMul(partial_result, lhs, rhs);

    gaugeAssign(lhs, partial_result);
    copyGauge (rhs, second_item);
    gaugeMul(partial_result, lhs, rhs);

    gaugeAssign(lhs, partial_result);
    copyGauge (rhs, third_item);
    gaugeTransposeConj(rhs);
    gaugeMul(partial_result, lhs, rhs);
  }

  __device__ void cloverCalculate(Complex* clover_ptr, const Point& point, int Lx, int Ly, int Lz, int Lt) {

    Complex clover_item[Ns * Nc * Ns * Nc / 2];
    Complex temp[Nc * Nc];
    Complex sum_gauge[Nc * Nc];

    setDst(point.getPointClover(clover_ptr, Lx, Ly, Lz, Lt));

    for (int i = 0; i < Ns * Nc * Ns * Nc / 2; i++) {
      clover_item[i].clear2Zero();
    }
    for (int i = 0; i < Ns; i++) {  // mu_
      for (int j = i+1; j < Ns; j++) {  //nu_
        for (int jj = 0; jj < Nc * Nc; jj++) {
          sum_gauge[jj].clear2Zero();
        }
        // F_1
        F_1(temp, point, i, j, Lx, Ly, Lz, Lt);
        gaugeAddAssign(sum_gauge, temp);

        // F_2
        F_2(temp, point, i, j, Lx, Ly, Lz, Lt);
        gaugeAddAssign(sum_gauge, temp);

        // F_3
        F_3(temp, point, i, j, Lx, Ly, Lz, Lt);
        gaugeAddAssign(sum_gauge, temp);

        // F_4
        F_4(temp, point, i, j, Lx, Ly, Lz, Lt);
        gaugeAddAssign(sum_gauge, temp);

        gaugeAssign(temp, sum_gauge);
        gaugeTransposeConj(temp);
        gaugeSubAssign(sum_gauge, temp);  // A-A.T.conj()

        tensorProductAddition(clover_item, sum_gauge, i, j, Complex(double(-1)/double(16), 0));
      }
    }

    for (int i = 0; i < Ns * Nc * Ns * Nc / 2; i++) {
      dst_[i] = clover_item[i];
    }
  }
};

__global__ void gpuCloverCalculate(void *fermion_out, void* invert_ptr, int Lx, int Ly, int Lz, int Lt, int parity) {
  assert(parity == 0 || parity == 1);
  // __shared__ double dst[BLOCK_SIZE * Ns * Nc * 2];
  __shared__ double shared_buffer[2 * Ns * Nc * BLOCK_SIZE];
  Lx >>= 1;
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Ly * Lx);
  int z = thread % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;

  Point p(x, y, z, t, parity);
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), Lx, Ly, Lz, Lt);
  Complex src_local[Ns * Nc]; // for GPU
  Complex dst_local[Ns * Nc]; // for GPU
  Complex invert_local[Ns * Nc * Ns * Nc / 2];

  // load src vector
  // loadVector(src_local, fermion_out, p, Lx, Ly, Lz, Lt);
  loadVectorBySharedMemory(static_cast<void*>(shared_buffer), static_cast<Complex*>(fermion_out), src_local);
  // loadCloverBySharedMemory(static_cast<Complex*>(invert_ptr), invert_local);
  Complex* invert_mem = p.getPointClover(static_cast<Complex*>(invert_ptr), Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns*Nc*Ns*Nc/2; i++) {
    invert_local[i] = invert_mem[i];
  }

  // A^{-1}dst
  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }
  for (int i = 0; i < Ns * Nc / 2; i++) {
    for (int j = 0; j < Ns * Nc / 2; j++) {
      dst_local[i] += invert_local[i*Ns*Nc/2+j] * src_local[j];
    }
  }
  for (int i = 0; i < Ns * Nc / 2; i++) {
    for (int j = 0; j < Ns * Nc / 2; j++) {
      dst_local[Ns*Nc/2+i] += invert_local[Ns*Nc*Ns*Nc/4 + i*Ns*Nc/2+j] * src_local[Ns*Nc/2+j];
    }
  }
  // end, and store dst
  // for (int i = 0; i < Ns * Nc; i++) {
  //   dst_ptr[i] = dst_local[i];
  // }
  storeVectorBySharedMemory(static_cast<void*>(shared_buffer), static_cast<Complex*>(fermion_out), dst_local);
}


__global__ void gpuClover(void* clover_ptr, void* invert_ptr, int Lx, int Ly, int Lz, int Lt, int parity, Complex* origin_gauge, Complex** shift_gauge, Complex** shift_shift_gauge) {

  assert(parity == 0 || parity == 1);
  Lx >>= 1;
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Ly * Lx);
  int z = thread % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;


  Clover clover(origin_gauge, shift_gauge, shift_shift_gauge);

  Point p(x, y, z, t, parity);

  Complex clover_local[(Ns * Nc * Ns * Nc) / 2];
  Complex invert_local[(Ns * Nc * Ns * Nc) / 2];
  Complex* invert_addr = p.getPointClover(static_cast<Complex*>(invert_ptr), Lx, Ly, Lz, Lt);

  clover.cloverCalculate(static_cast<Complex*>(clover_ptr), p, Lx, Ly, Lz, Lt);

  Complex* clover_addr = clover.getDst();


  for (int i = 0; i < Ns * Nc * Ns * Nc / 2; i++) {
    clover_local[i] = clover_addr[i];
  }
  // generate A = 1 + T     TODO: optimize
  for (int i = 0; i < Ns*Nc/2; i++) {
    clover_local[                i*Ns*Nc/2 + i] += Complex(1, 0);
    clover_local[Ns*Nc*Ns*Nc/4 + i*Ns*Nc/2 + i] += Complex(1, 0);
  }
  // store A = 1+T
  for (int i = 0; i < Ns * Nc * Ns * Nc / 2; i++) {
    clover_addr[i] = clover_local[i];
  } // added on 2023, 10, 18, by wjc

  // invert A_{oo}
  inverseMatrix(clover_local, invert_local);
  // invert A_{ee}
  inverseMatrix(clover_local + Ns*Nc*Ns*Nc/4, invert_local + Ns*Nc*Ns*Nc/4);
  // store invert vector A_{-1}
  for (int i = 0; i < Ns * Nc * Ns * Nc/2; i++) {
    invert_addr[i] = invert_local[i];
  }

  // storeCloverBySharedMemory(invert_addr, invert_local);
}


extern int grid_x;
extern int grid_y;
extern int grid_z;
extern int grid_t;

static int new_clover_computation;
static bool clover_prepared;
static bool clover_allocated;
static void* clover_matrix;
static void* invert_matrix;

static __global__ void mpiDslash(void *gauge, void *fermion_in, void *fermion_out,int Lx, int Ly, int Lz, int Lt, int parity, int grid_x, int grid_y, int grid_z, int grid_t, void* flag_ptr) {
  assert(parity == 0 || parity == 1);

  __shared__ double shared_buffer[BLOCK_SIZE * Ns * Nc * 2];
  Lx >>= 1;

  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Ly * Lx);
  int z = thread % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;

  int coord_boundary;
  Complex flag = *(static_cast<Complex*>(flag_ptr));



  Point p(x, y, z, t, parity);
  Point move_point;
  Complex u_local[Nc * Nc];   // for GPU
  Complex src_local[Ns * Nc]; // for GPU
  Complex dst_local[Ns * Nc]; // for GPU
  Complex temp;
  int eo = (y+z+t) & 0x01;

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

  // \mu = 1
  loadGauge(u_local, gauge, 0, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 0, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  // x front    x == Lx-1 && parity != eo
  coord_boundary = (grid_x > 1 && x == Lx-1 && parity != eo) ? Lx-1 : Lx;
  if (x < coord_boundary) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp = (src_local[0 * Nc + j] - flag * src_local[3 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
        dst_local[0 * Nc + i] += temp;
        dst_local[3 * Nc + i] += temp * Complex(0, 1) * flag;
        // second row vector with col vector
        temp = (src_local[1 * Nc + j] - flag * src_local[2 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
        dst_local[1 * Nc + i] += temp;
        dst_local[2 * Nc + i] += temp * Complex(0, 1) * flag;
      }
    }
  }

  // x back  x==0 && parity == eo
  move_point = p.move(BACK, 0, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, 0, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  coord_boundary = (grid_x > 1 && x==0 && parity == eo) ? 1 : 0;
  if (x >= coord_boundary) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp = (src_local[0 * Nc + j] + flag * src_local[3 * Nc + j] * Complex(0, 1)) *
              u_local[j * Nc + i].conj(); // transpose and conj
        dst_local[0 * Nc + i] += temp;
        dst_local[3 * Nc + i] += temp * Complex(0, -1) * flag;
        // second row vector with col vector
        temp = (src_local[1 * Nc + j] + flag * src_local[2 * Nc + j] * Complex(0, 1)) *
              u_local[j * Nc + i].conj(); // transpose and conj
        dst_local[1 * Nc + i] += temp;
        dst_local[2 * Nc + i] += temp * Complex(0, -1) * flag;
      }
    }
  }

  // \mu = 2
  // y front
  loadGauge(u_local, gauge, 1, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 1, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  coord_boundary = (grid_y > 1) ? Ly-1 : Ly;
  if (y < coord_boundary) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp = (src_local[0 * Nc + j] + flag * src_local[3 * Nc + j]) * u_local[i * Nc + j];
        dst_local[0 * Nc + i] += temp;
        dst_local[3 * Nc + i] += temp * flag;
        // second row vector with col vector
        temp = (src_local[1 * Nc + j] - flag * src_local[2 * Nc + j]) * u_local[i * Nc + j];
        dst_local[1 * Nc + i] += temp;
        dst_local[2 * Nc + i] += -temp * flag;
      }
    }
  }

  // y back
  move_point = p.move(BACK, 1, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, 1, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_y > 1) ? 1 : 0;
  if (y >= coord_boundary) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp = (src_local[0 * Nc + j] - flag * src_local[3 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
        dst_local[0 * Nc + i] += temp;
        dst_local[3 * Nc + i] += -temp * flag;
        // second row vector with col vector
        temp = (src_local[1 * Nc + j] + flag * src_local[2 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
        dst_local[1 * Nc + i] += temp;
        dst_local[2 * Nc + i] += temp * flag;
      }
    }
  }

  // \mu = 3
  // z front
  loadGauge(u_local, gauge, 2, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 2, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  coord_boundary = (grid_z > 1) ? Lz-1 : Lz;
  if (z < coord_boundary) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp = (src_local[0 * Nc + j] - flag * src_local[2 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
        dst_local[0 * Nc + i] += temp;
        dst_local[2 * Nc + i] += temp * Complex(0, 1) * flag;
        // second row vector with col vector
        temp = (src_local[1 * Nc + j] + flag * src_local[3 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
        dst_local[1 * Nc + i] += temp;
        dst_local[3 * Nc + i] += temp * Complex(0, -1) * flag;
      }
    }
  }

  // z back
  move_point = p.move(BACK, 2, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, 2, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_z > 1) ? 1 : 0;
  if (z >= coord_boundary) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp = (src_local[0 * Nc + j] + flag * src_local[2 * Nc + j] * Complex(0, 1)) *
              u_local[j * Nc + i].conj(); // transpose and conj
        dst_local[0 * Nc + i] += temp;
        dst_local[2 * Nc + i] += temp * Complex(0, -1) * flag;
        // second row vector with col vector
        temp = (src_local[1 * Nc + j] - flag * src_local[3 * Nc + j] * Complex(0, 1)) *
              u_local[j * Nc + i].conj(); // transpose and conj
        dst_local[1 * Nc + i] += temp;
        dst_local[3 * Nc + i] += temp * Complex(0, 1) * flag;
      }
    }
  }

  // t: front
  loadGauge(u_local, gauge, 3, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 3, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_t > 1) ? Lt-1 : Lt;
  if (t < coord_boundary) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp = (src_local[0 * Nc + j] - flag * src_local[2 * Nc + j]) * u_local[i * Nc + j];
        dst_local[0 * Nc + i] += temp;
        dst_local[2 * Nc + i] += -temp * flag;
        // second row vector with col vector
        temp = (src_local[1 * Nc + j] - flag * src_local[3 * Nc + j]) * u_local[i * Nc + j];
        dst_local[1 * Nc + i] += temp;
        dst_local[3 * Nc + i] += -temp * flag;
      }
    }
  }
  // t: back
  move_point = p.move(BACK, 3, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, 3, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_t > 1) ? 1 : 0;
  if (t >= coord_boundary) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp = (src_local[0 * Nc + j] + flag * src_local[2 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
        dst_local[0 * Nc + i] += temp;
        dst_local[2 * Nc + i] += temp * flag;
        // second row vector with col vector
        temp = (src_local[1 * Nc + j] + flag * src_local[3 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
        dst_local[1 * Nc + i] += temp;
        dst_local[3 * Nc + i] += temp * flag;
      }
    }
  }

  // store result
  storeVectorBySharedMemory(static_cast<void*>(shared_buffer), static_cast<Complex*>(fermion_out), dst_local);
}




CloverDslash::CloverDslash(DslashParam& param) : Dslash(param){
  int Lx = dslashParam_->Lx;
  int Ly = dslashParam_->Ly;
  int Lz = dslashParam_->Lz;
  int Lt = dslashParam_->Lt;

  int vol = Lx * Ly * Lz * Lt >> 1;
  dim3 gridDim(vol / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);

  int parity = 0;
  if (!clover_allocated) {
    checkCudaErrors(cudaMalloc(&clover_matrix, sizeof(Complex) * Ns * Nc * Ns * Nc / 2 * Lx * Ly * Lz * Lt));
    checkCudaErrors(cudaMalloc(&invert_matrix, sizeof(Complex) * Ns * Nc * Ns * Nc / 2 * Lx * Ly * Lz * Lt));
    clover_allocated = true;
  }

  if (!clover_prepared) {
    mpi_comm->allocateGaugeBuffer();
    mpi_comm->prepareGauge();

    void* origin_gauge = dslashParam_->gauge;
    Complex** shift_gauge = mpi_comm->getShiftGauge();
    Complex** shift_shift_gauge = mpi_comm->getShiftShiftGauge();
    Complex** d_shift_gauge;
    Complex** d_shift_shift_gauge;
    checkCudaErrors(cudaMalloc(&d_shift_gauge, sizeof(Complex) * Nd * 2));
    checkCudaErrors(cudaMalloc(&d_shift_shift_gauge, sizeof(Complex) * 6 * 4));
    checkCudaErrors(cudaMemcpy(d_shift_gauge, shift_gauge, sizeof(Complex) * Nd * 2, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_shift_shift_gauge, shift_shift_gauge, sizeof(Complex) * 6 * 4, cudaMemcpyHostToDevice));
    parity = 0;

    void *args[] = {&clover_matrix, &invert_matrix, &Lx, &Ly, &Lz, &Lt, &parity, &origin_gauge, &d_shift_gauge, &d_shift_shift_gauge};
    checkCudaErrors(cudaLaunchKernel((void *)gpuClover, gridDim, blockDim, args));
    checkCudaErrors(cudaDeviceSynchronize());

    parity = 1;
    checkCudaErrors(cudaLaunchKernel((void *)gpuClover, gridDim, blockDim, args));
    checkCudaErrors(cudaDeviceSynchronize());
    
    checkCudaErrors(cudaFree(d_shift_gauge));
    checkCudaErrors(cudaFree(d_shift_shift_gauge));
    if (new_clover_computation) {
      clover_prepared = false;
    } else {
      clover_prepared = true;
    }
    mpi_comm->freeGaugeBuffer();
  }
}

void CloverDslash::inverseCloverResult(void* p_fermion_out, void* p_invert_matrix, int Lx, int Ly, int Lz, int Lt, int parity) {
  int space = Lx * Ly * Lz * Lt >> 1;
  dim3 gridDim(space / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);

  void *args[] = {&p_fermion_out, &p_invert_matrix, &Lx, &Ly, &Lz, &Lt, &parity};
  checkCudaErrors(cudaLaunchKernel((void *)gpuCloverCalculate, gridDim, blockDim, args));
  checkCudaErrors(cudaDeviceSynchronize());
}

void CloverDslash::cloverResult(void* p_fermion_out, void* p_clover_matrix, int Lx, int Ly, int Lz, int Lt, int parity) {
  int space = Lx * Ly * Lz * Lt >> 1;
  dim3 gridDim(space / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);

  void *args[] = {&p_fermion_out, &p_clover_matrix, &Lx, &Ly, &Lz, &Lt, &parity};
  checkCudaErrors(cudaLaunchKernel((void *)gpuCloverCalculate, gridDim, blockDim, args));
  checkCudaErrors(cudaDeviceSynchronize());
}

void CloverDslash::calculateDslash(int invert_flag) {
  int Lx = dslashParam_->Lx;
  int Ly = dslashParam_->Ly;
  int Lz = dslashParam_->Lz;
  int Lt = dslashParam_->Lt;
  int parity = dslashParam_->parity;
  Complex h_flag;
  if (invert_flag == 0) {
    h_flag = Complex(1, 0);
    // h_flag = Complex(0, 0); //debug
  } else {
    h_flag = Complex(-1, 0);
  }
  Complex* d_flag;

  checkCudaErrors(cudaMalloc(&d_flag, sizeof(Complex)));
  checkCudaErrors(cudaMemcpy(d_flag, &h_flag, sizeof(Complex), cudaMemcpyHostToDevice));

  int space = Lx * Ly * Lz * Lt >> 1;
  dim3 gridDim(space / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);

  checkCudaErrors(cudaDeviceSynchronize());
  // checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

  mpi_comm->preDslash(dslashParam_->fermion_in, parity, invert_flag);

  auto start = std::chrono::high_resolution_clock::now();
  void *args[] = {&dslashParam_->gauge, &dslashParam_->fermion_in, &dslashParam_->fermion_out, &Lx, &Ly, &Lz, &Lt, &parity, &grid_x, &grid_y, &grid_z, &grid_t, &d_flag};

  checkCudaErrors(cudaMemcpy(d_flag, &h_flag, sizeof(Complex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaLaunchKernel((void *)mpiDslash, gridDim, blockDim, args));
  checkCudaErrors(cudaDeviceSynchronize());
  // boundary calculate
  mpi_comm->postDslash(dslashParam_->fermion_out, parity, invert_flag);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  // printf("total time: (without malloc free memcpy) : %.9lf sec\n", double(duration) / 1e9);
  inverseCloverResult(dslashParam_->fermion_out, invert_matrix, Lx, Ly, Lz, Lt, parity);
  checkCudaErrors(cudaFree(d_flag));
}






// void callCloverDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag) {
//   DslashParam dslash_param(fermion_in, fermion_out, gauge, param, parity);
//   CloverDslash dslash_solver(dslash_param);
//   dslash_solver.calculateDslash(invert_flag);
// }


void invertCloverDslash (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int invert_flag) {
  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol / 2;
  for (int parity = 0; parity < 2; parity++) {
    void* half_fermion_in = static_cast<void*>(static_cast<Complex*>(fermion_in) + (1 - parity) * half_vol * Ns * Nc);
    void* half_fermion_out = static_cast<void*>(static_cast<Complex*>(fermion_out) + parity * half_vol * Ns * Nc);

    DslashParam dslash_param(half_fermion_in, half_fermion_out, gauge, param, parity);
    CloverDslash dslash_solver(dslash_param);
    //   inverseCloverResult(dslashParam_->fermion_out, invert_matrix, Lx, Ly, Lz, Lt, parity);

    dslash_solver.inverseCloverResult(half_fermion_out, invert_matrix, Lx, Ly, Lz, Lt, parity); // Clover
  }
}
void callCloverDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag) {
  DslashParam dslash_param(fermion_in, fermion_out, gauge, param, parity);
  CloverDslash dslash_solver(dslash_param);
  dslash_solver.calculateDslash(invert_flag);
}


void preCloverDslash (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int invert_flag) {
  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol / 2;
  for (int parity = 0; parity < 2; parity++) {
    void* half_fermion_in = static_cast<void*>(static_cast<Complex*>(fermion_in) + (1 - parity) * half_vol * Ns * Nc);
    void* half_fermion_out = static_cast<void*>(static_cast<Complex*>(fermion_out) + parity * half_vol * Ns * Nc);

    DslashParam dslash_param(half_fermion_in, half_fermion_out, gauge, param, parity);
    CloverDslash dslash_solver(dslash_param);
    //   inverseCloverResult(dslashParam_->fermion_out, invert_matrix, Lx, Ly, Lz, Lt, parity);

    dslash_solver.inverseCloverResult(half_fermion_out, clover_matrix, Lx, Ly, Lz, Lt, parity); // Clover
  }
}



void fullCloverDslashOneRound (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int invert_flag) {
  // A = (1 + T)     A_{-1} Dslash
  cloverDslashOneRound(fermion_out, fermion_in, gauge, param, invert_flag);

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol / 2;

  Complex h_kappa(1, 0);
  Complex h_coeff(1, 0);
  Complex* d_coeff;
  Complex* d_kappa;
  checkCudaErrors(cudaMalloc(&d_coeff, sizeof(Complex)));
  checkCudaErrors(cudaMalloc(&d_kappa, sizeof(Complex)));

  // h_kappa *= Complex(-1, 0);
  h_kappa = h_kappa * Complex(-1, 0);

  checkCudaErrors(cudaMemcpy(d_coeff, &h_coeff, sizeof(Complex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_kappa, &h_kappa, sizeof(Complex), cudaMemcpyHostToDevice));
  mpi_comm->interprocess_sax_barrier(fermion_out, d_kappa, vol);    // -kappa * left
  mpi_comm->interprocess_saxpy_barrier(fermion_in, fermion_out, d_coeff, vol);  // src + kappa * dst = dst    coeff=1

  for (int parity = 0; parity < 2; parity++) {
    void* half_fermion_in = static_cast<void*>(static_cast<Complex*>(fermion_in) + (1 - parity) * half_vol * Ns * Nc);
    void* half_fermion_out = static_cast<void*>(static_cast<Complex*>(fermion_out) + parity * half_vol * Ns * Nc);

    DslashParam dslash_param(half_fermion_in, half_fermion_out, gauge, param, parity);
    CloverDslash dslash_solver(dslash_param);

    dslash_solver.cloverResult(half_fermion_out, clover_matrix, Lx, Ly, Lz, Lt, parity); // Clover
  }

  checkCudaErrors(cudaFree(d_coeff));
  checkCudaErrors(cudaFree(d_kappa));

}

// this fermion in is the start addr of all
void cloverDslashOneRound(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int invert_flag) {

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol / 2;


  // Dslash when invert_flag is 0, else if 1---->Dslash dagger
  for (int parity = 0; parity < 2; parity++) {
    void* half_fermion_in = static_cast<void*>(static_cast<Complex*>(fermion_in) + (1 - parity) * half_vol * Ns * Nc);
    void* half_fermion_out = static_cast<void*>(static_cast<Complex*>(fermion_out) + parity * half_vol * Ns * Nc);
    callCloverDslash(half_fermion_out, half_fermion_in, gauge, param, parity, invert_flag);
  }

  // checkCudaErrors(cudaFree(d_coeff));
  // checkCudaErrors(cudaFree(d_kappa));
}



void newFullCloverDslashOneRound (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int invert_flag) {
  // A = (1 + T)     A_{-1} Dslash
  cloverDslashOneRound(fermion_out, fermion_in, gauge, param, invert_flag);

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  // int half_vol = vol / 2;

  Complex h_kappa(1, 0);
  Complex h_coeff(1, 0);
  Complex* d_coeff;
  Complex* d_kappa;
  checkCudaErrors(cudaMalloc(&d_coeff, sizeof(Complex)));
  checkCudaErrors(cudaMalloc(&d_kappa, sizeof(Complex)));

  // h_kappa *= Complex(-1, 0);
  h_kappa = h_kappa * Complex(-1, 0);

  checkCudaErrors(cudaMemcpy(d_coeff, &h_coeff, sizeof(Complex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_kappa, &h_kappa, sizeof(Complex), cudaMemcpyHostToDevice));
  mpi_comm->interprocess_sax_barrier(fermion_out, d_kappa, vol);    // -kappa * left
  mpi_comm->interprocess_saxpy_barrier(fermion_in, fermion_out, d_coeff, vol);  // src + kappa * dst = dst    coeff=1

  checkCudaErrors(cudaFree(d_coeff));
  checkCudaErrors(cudaFree(d_kappa));

}

void MmV_one_round (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, void* temp) {
  // fullCloverDslashOneRound(temp, fermion_in, gauge, param, 0); // Dslash vec
  // fullCloverDslashOneRound(fermion_out, temp, gauge, param, 1); // Dslash^\dagger Dslash vec
  newFullCloverDslashOneRound(temp, fermion_in, gauge, param, 0); // Dslash vec
  newFullCloverDslashOneRound(fermion_out, temp, gauge, param, 1); // Dslash^\dagger Dslash vec
}


__attribute__((constructor)) void initialize() {
  // mpi_comm = nullptr;
  clover_prepared = false;
  new_clover_computation = 0;
  clover_allocated = false;
}

