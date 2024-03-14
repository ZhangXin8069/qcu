
#include "qcu_communicator.cuh"
#include "qcu_point.cuh"
#include "qcu_complex_computation.cuh"
#include "qcu.h"
// #define DEBUG

// static / global variables
int grid_x;
int grid_y;
int grid_z;
int grid_t;

static int process_rank;
static int process_num;
// static cudaStream_t stream[Nd][2];  // Nd means dims, 2 means FRONT BACK
// end
static __device__ inline void reconstructSU3(Complex *su3)
{
  su3[6] = (su3[1] * su3[5] - su3[2] * su3[4]).conj();
  su3[7] = (su3[2] * su3[3] - su3[0] * su3[5]).conj();
  su3[8] = (su3[0] * su3[4] - su3[1] * su3[3]).conj();
}
static __device__ void copyGauge (Complex* dst, Complex* src) {
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    dst[i] = src[i];
  }
  reconstructSU3(dst);
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
__global__ void DslashTransferFrontX(void *gauge, void *fermion_in,int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer, void* flag_ptr) {
  // 前传传结果
  int sub_Lx = (Lx >> 1);
  int sub_Ly = (Ly >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Ly);
  int z = thread % (Lz * sub_Ly) / sub_Ly;
  int sub_y = thread % sub_Ly;
  Complex flag = *(static_cast<Complex*>(flag_ptr));

  int new_even_odd = (z+t) & 0x01;
  Point p(sub_Lx-1, 2 * sub_y + (new_even_odd == 1-parity), z, t, 1-parity);

  Complex* dst_ptr;

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];
  Complex temp;
  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, X_DIRECTION, p, sub_Lx, Ly, Lz, Lt);

  // even save to even, odd save to even
  dst_ptr = send_buffer + thread*Ns*Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

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

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }
}

__global__ void DslashTransferBackX(void *fermion_in, int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer) {
  // 后传传向量
  int sub_Lx = (Lx >> 1);
  int sub_Ly = (Ly >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;   // 注意这里乘以2
  int t = thread / (Lz * sub_Ly);
  int z = thread % (Lz * sub_Ly) / sub_Ly;
  int sub_y = thread % sub_Ly;

  int new_even_odd = (z+t) & 0x01;
  Point p(0, 2 * sub_y + (new_even_odd != 1-parity), z, t, 1-parity);

  Complex* dst_ptr;
  Complex src_local[Ns * Nc];

  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);

  dst_ptr = send_buffer + thread * Ns * Nc;
  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = src_local[i];
  }
}
__global__ void DslashTransferFrontY(void *gauge, void *fermion_in,int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer, void* flag_ptr) {
  // 前传传结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Lx);
  int z = thread % (Lz * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, Ly-1, z, t, 1-parity);
  Complex flag = *(static_cast<Complex*>(flag_ptr));

  Complex* dst_ptr;

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];
  Complex temp;
  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, Y_DIRECTION, p, sub_Lx, Ly, Lz, Lt);

  // even save to even, odd save to even
  dst_ptr = send_buffer + thread*Ns*Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

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

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }
}
__global__ void DslashTransferBackY(void *fermion_in, int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer) {
  // 后传传向量
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Lx);
  int z = thread % (Lz * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, 0, z, t, 1-parity);

  Complex* dst_ptr;
  Complex src_local[Ns * Nc];

  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);

  dst_ptr = send_buffer + thread * Ns * Nc;
  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = src_local[i];
  }
}
// DslashTransferFrontZ: DONE
__global__ void DslashTransferFrontZ(void *gauge, void *fermion_in,int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer, void* flag_ptr) {
  // 前传传结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, Lz-1, t, 1-parity);
  Complex flag = *(static_cast<Complex*>(flag_ptr));

  Complex* dst_ptr;

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];
  Complex temp;
  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, Z_DIRECTION, p, sub_Lx, Ly, Lz, Lt);

  // even save to even, odd save to even
  dst_ptr = send_buffer + thread*Ns*Nc;//((z * Ly + y) * sub_Lx + x) * Ns * Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

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


  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }

}
// DslashTransferBackZ: Done
__global__ void DslashTransferBackZ(void *fermion_in, int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer) {
  // 后传传向量
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, 0, t, 1-parity);

  Complex* dst_ptr;
  Complex src_local[Ns * Nc];

  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);

  dst_ptr = send_buffer + thread * Ns * Nc;
  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = src_local[i];
  }
}

// DslashTransferFrontT: Done
__global__ void DslashTransferFrontT(void *gauge, void *fermion_in,int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer, void* flag_ptr){
  // 前传传结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int z = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, z, Lt-1, 1-parity);
  Complex flag = *(static_cast<Complex*>(flag_ptr));
#ifdef DEBUG
  if(thread == 0) {
    printf(RED"%lf, %lf\n", flag.real(), flag.imag());
    printf(CLR"");
  }
#endif
  Complex* dst_ptr;

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];
  Complex temp;
  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, T_DIRECTION, p, sub_Lx, Ly, Lz, Lt);

  // even save to even, odd save to even
  dst_ptr = send_buffer + thread*Ns*Nc;//((z * Ly + y) * sub_Lx + x) * Ns * Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }
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

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }


}
// DslashTransferBackT: Done
__global__ void DslashTransferBackT(void *fermion_in, int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer) {
  // 后传传向量
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int z = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, z, 0, 1-parity);

  Complex* dst_ptr;
  Complex src_local[Ns * Nc];

  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);

  dst_ptr = send_buffer + thread * Ns * Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = src_local[i];
  }
}

__global__ void calculateBackBoundaryX(void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer) {
  int sub_Lx = (Lx >> 1);
  int sub_Ly = (Ly >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Ly);
  int z = thread % (Lz * sub_Ly) / sub_Ly;
  int sub_y = thread % sub_Ly;
  
  int new_even_odd = (z+t) & 0x01;  // %2
  Point p(0, 2 * sub_y + (new_even_odd != parity), z, t, parity);

  Complex* src_ptr;
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex dst_local[Ns * Nc];
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
  src_ptr = recv_buffer + thread*Ns*Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src_ptr[i];
  }
  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i] += src_local[i];
  }

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }

}
__global__ void calculateFrontBoundaryX(void* gauge, void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer, void* flag_ptr) {
  int sub_Lx = (Lx >> 1);
  int sub_Ly = (Ly >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;   // 注意这里乘以2
  int t = thread / (Lz * sub_Ly);
  int z = thread % (Lz * sub_Ly) / sub_Ly;
  int sub_y = thread % sub_Ly;
  Complex flag = *(static_cast<Complex*>(flag_ptr));

  int new_even_odd = (z+t) & 0x01;  // %2
  Point p(sub_Lx-1, 2 * sub_y + (new_even_odd == parity), z, t, parity);

  Complex temp;
  Complex* src_ptr;
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];

  loadGauge(u_local, gauge, X_DIRECTION, p, sub_Lx, Ly, Lz, Lt);
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);

  src_ptr = recv_buffer + thread * Ns * Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src_ptr[i];
  }
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

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }
}

__global__ void calculateBackBoundaryY(void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer) {
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Lx);
  int z = thread % (Lz * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, 0, z, t, parity);

  Complex* src_ptr;
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex dst_local[Ns * Nc];
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
  src_ptr = recv_buffer + thread*Ns*Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src_ptr[i];
  }
  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i] += src_local[i];
  }

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }
}

__global__ void calculateFrontBoundaryY(void* gauge, void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer, void* flag_ptr) {
  // 后接接结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Lx);
  int z = thread % (Lz * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, Ly-1, z, t, parity);
  Complex flag = *(static_cast<Complex*>(flag_ptr));
  Complex temp;
  Complex* src_ptr;
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];

  loadGauge(u_local, gauge, Y_DIRECTION, p, sub_Lx, Ly, Lz, Lt);
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);

  src_ptr = recv_buffer + thread * Ns * Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src_ptr[i];
  }

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

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }
}
__global__ void calculateBackBoundaryZ(void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer) {
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, 0, t, parity);

  Complex* src_ptr;
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex dst_local[Ns * Nc];
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
  src_ptr = recv_buffer + thread*Ns*Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src_ptr[i];
  }
  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i] += src_local[i];
  }

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }
}

__global__ void calculateFrontBoundaryZ(void* gauge, void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer, void* flag_ptr) {
  // 后接接结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, Lz-1, t, parity);
  Complex flag = *(static_cast<Complex*>(flag_ptr));

  Complex temp;
  Complex* src_ptr;
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];

  loadGauge(u_local, gauge, Z_DIRECTION, p, sub_Lx, Ly, Lz, Lt);
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);

  src_ptr = recv_buffer + thread * Ns * Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src_ptr[i];
  }

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

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }

}

__global__ void calculateBackBoundaryT(void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer) {
  // 后接接结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int z = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, z, 0, parity);

  Complex* src_ptr;
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex dst_local[Ns * Nc];
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
  src_ptr = recv_buffer + thread*Ns*Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src_ptr[i];
  }
  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i] += src_local[i];
  }

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }
}

__global__ void calculateFrontBoundaryT(void* gauge, void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer, void* flag_ptr) {
  // 后接接结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int z = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, z, Lt-1, parity);
  Complex flag = *(static_cast<Complex*>(flag_ptr));

  Complex temp;
  Complex* src_ptr;
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];

  loadGauge(u_local, gauge, 3, p, sub_Lx, Ly, Lz, Lt);
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);

  src_ptr = recv_buffer + thread * Ns * Nc;//((z * Ly + y) * sub_Lx + x) * Ns * Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src_ptr[i];
  }

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

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }
}

__global__ void sendGaugeBoundaryToBufferX(void* origin_gauge, void* front_buffer, void* back_buffer, int Lx, int Ly, int Lz, int Lt) {
  int parity;
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Ly);
  int z = thread % (Lz * Ly) / Ly;
  int y = thread % Ly;

  int x;
  int sub_x;
  int single_gauge = Ly * Lz * Lt * Nc * Nc;
  Complex* src;
  Complex* dst;

  // forward
  x = Lx - 1;
  sub_x = x >> 1;
  parity = (x + y + z + t) & 0x01;
  Point f_p(sub_x, y, z, t, parity);
  for (int i = 0; i < Nd; i++) {
    src = f_p.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
    dst = static_cast<Complex*>(front_buffer) + i * single_gauge + ((t * Lz + z) * Ly + y) * Nc * Nc;
    copyGauge(dst, src);
  }

  // backward
  x = 0;
  sub_x = x >> 1;
  parity = (x + y + z + t) & 0x01;
  f_p = Point(sub_x, y, z, t, parity);
  for (int i = 0; i < Nd; i++) {
    src = f_p.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
    dst = static_cast<Complex*>(back_buffer) + i * single_gauge + ((t * Lz + z) * Ly + y) * Nc * Nc;
    copyGauge(dst, src);
  }
}

__global__ void sendGaugeBoundaryToBufferY(void* origin_gauge, void* front_buffer, void* back_buffer, int Lx, int Ly, int Lz, int Lt) {
  int parity;
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Lx);
  int z = thread % (Lz * Lx) / Lx;
  int x = thread % Lx;
  int sub_x = x >> 1;
  int single_gauge = Lx * Lz * Lt * Nc * Nc;
  int y;
  Complex* src;
  Complex* dst;

  // forward
  y = Ly - 1;
  parity = (x + y + z + t) & 0x01;
  Point f_p(sub_x, y, z, t, parity);
  for (int i = 0; i < Nd; i++) {
    src = f_p.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
    dst = static_cast<Complex*>(front_buffer) + i * single_gauge + ((t * Lz + z) * Lx + x) * Nc * Nc;
    copyGauge(dst, src);

  }

  // backward
  y = 0;
  parity = (x + y + z + t) & 0x01;
  f_p = Point(sub_x, y, z, t, parity);
  for (int i = 0; i < Nd; i++) {
    src = f_p.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
    dst = static_cast<Complex*>(back_buffer) + i * single_gauge + ((t * Lz + z) * Lx + x) * Nc * Nc;
    copyGauge(dst, src);
  }
}
__global__ void sendGaugeBoundaryToBufferZ(void* origin_gauge, void* front_buffer, void* back_buffer, int Lx, int Ly, int Lz, int Lt) {
  int parity;
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;
  int sub_x = x >> 1;
  int single_gauge = Lx * Ly * Lt * Nc * Nc;
  int z;
  Complex* src;
  Complex* dst;

  // forward
  z = Lz - 1;
  parity = (x + y + z + t) & 0x01;
  Point f_p(sub_x, y, z, t, parity);
  for (int i = 0; i < Nd; i++) {
    src = f_p.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
    dst = static_cast<Complex*>(front_buffer) + i * single_gauge + ((t * Ly + y) * Lx + x) * Nc * Nc;
    copyGauge(dst, src);

  }

  // backward
  z = 0;
  parity = (x + y + z + t) & 0x01;
  f_p = Point(sub_x, y, z, t, parity);
  for (int i = 0; i < Nd; i++) {
    src = f_p.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
    dst = static_cast<Complex*>(back_buffer) + i * single_gauge + ((t * Ly + y) * Lx + x) * Nc * Nc;
    copyGauge(dst, src);
  }
}
__global__ void sendGaugeBoundaryToBufferT(void* origin_gauge, void* front_buffer, void* back_buffer, int Lx, int Ly, int Lz, int Lt) {
  int parity;
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int z = thread / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;
  int sub_x = x >> 1;
  int single_gauge = Lx * Ly * Lz * Nc * Nc;
  int t;
  Complex* src;
  Complex* dst;

  // forward
  t = Lt - 1;
  parity = (x + y + z + t) & 0x01;
  Point f_p(sub_x, y, z, t, parity);
  for (int i = 0; i < Nd; i++) {
    src = f_p.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
    dst = static_cast<Complex*>(front_buffer) + i * single_gauge + ((z * Ly + y) * Lx + x) * Nc * Nc;
    copyGauge(dst, src);

  }

  // backward
  t = 0;
  parity = (x + y + z + t) & 0x01;
  f_p = Point(sub_x, y, z, t, parity);
  for (int i = 0; i < Nd; i++) {
    src = f_p.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
    dst = static_cast<Complex*>(back_buffer) + i * single_gauge + ((z * Ly + y) * Lx + x) * Nc * Nc;
    copyGauge(dst, src);
  }
}

__global__ void shiftGaugeX(void* origin_gauge, void* front_shift_gauge, void* back_shift_gauge, void* front_boundary, void* back_boundary, int Lx, int Ly, int Lz, int Lt) {
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Ly * Lx);
  int z = thread % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;

  int sub_x = (x >> 1);
  int single_gauge = Ly * Lz * Lt * Nc * Nc;
  int parity = (x + y + z + t) & 0x01;

  Complex* src;
  Complex* dst;

  Point point(sub_x, y, z, t, parity);
  Point move_point;

  // backward-shift
  if (x < Lx - 1) {
    move_point = Point((x+1) >> 1, y, z, t, 1-parity);
    for (int i = 0; i < Nd; i++) {  // i means four dims
      src = point.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
      dst = move_point.getPointGauge(static_cast<Complex*>(back_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  } else {
    move_point = Point(0, y, z, t, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = static_cast<Complex*>(back_boundary) + i * single_gauge + ((t * Lz + z) * Ly + y) * Nc * Nc;
      dst = move_point.getPointGauge(static_cast<Complex*>(back_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  }

  // forward-shift
  if (x > 0) {
    move_point = Point((x-1) >> 1, y, z, t, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = point.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
      dst = move_point.getPointGauge(static_cast<Complex*>(front_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  } else {
    // backward boundary to 0 line
    move_point = Point((Lx-1)>>1, y, z, t, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = static_cast<Complex*>(front_boundary) + i * single_gauge + ((t * Lz + z) * Ly + y) * Nc * Nc;
      dst = move_point.getPointGauge(static_cast<Complex*>(front_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  }
}

__global__ void shiftGaugeY(void* origin_gauge, void* front_shift_gauge, void* back_shift_gauge, void* front_boundary, void* back_boundary, int Lx, int Ly, int Lz, int Lt) {
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Ly * Lx);
  int z = thread % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;

  int sub_x = (x >> 1);
  int single_gauge = Lx * Lz * Lt * Nc * Nc;
  int parity = (x + y + z + t) & 0x01;

  Complex* src;
  Complex* dst;

  Point point(sub_x, y, z, t, parity);
  Point move_point;

  // backward-shift
  if (y < Ly - 1) {
    move_point = Point(sub_x, y+1, z, t, 1-parity);
    for (int i = 0; i < Nd; i++) {  // i means four dims
      src = point.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
      dst = move_point.getPointGauge(static_cast<Complex*>(back_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  } else {
    // forward boundary to Lt-1 line
    move_point = Point(sub_x, 0, z, t, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = static_cast<Complex*>(back_boundary) + i * single_gauge + ((t * Lz + z) * Lx + x) * Nc * Nc;
      dst = move_point.getPointGauge(static_cast<Complex*>(back_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  }

  // forward-shift
  // if (t < Lt-1) {
  if (y > 0) {
    move_point = Point(sub_x, y-1, z, t, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = point.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
      dst = move_point.getPointGauge(static_cast<Complex*>(front_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  } else {
    // backward boundary to 0 line
    move_point = Point(sub_x, Ly-1, z, t, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = static_cast<Complex*>(front_boundary) + i * single_gauge + ((t * Lz + z) * Lx + x) * Nc * Nc;
      dst = move_point.getPointGauge(static_cast<Complex*>(front_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  }
}

__global__ void shiftGaugeZ(void* origin_gauge, void* front_shift_gauge, void* back_shift_gauge, void* front_boundary, void* back_boundary, int Lx, int Ly, int Lz, int Lt) {
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Ly * Lx);
  int z = thread % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;

  int sub_x = (x >> 1);
  int single_gauge = Lx * Ly * Lt * Nc * Nc;
  int parity = (x + y + z + t) & 0x01;

  Complex* src;
  Complex* dst;

  Point point(sub_x, y, z, t, parity);
  Point move_point;

  // backward-shift
  if (z < Lz - 1) {
    move_point = Point(sub_x, y, z+1, t, 1-parity);
    for (int i = 0; i < Nd; i++) {  // i means four dims
      src = point.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
      dst = move_point.getPointGauge(static_cast<Complex*>(back_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  } else {
    // forward boundary to Lt-1 line
    // move_point = Point(sub_x, y, z, Lt-1, 1-parity);
    move_point = Point(sub_x, y, 0, t, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = static_cast<Complex*>(back_boundary) + i * single_gauge + ((t * Ly + y) * Lx + x) * Nc * Nc;
      dst = move_point.getPointGauge(static_cast<Complex*>(back_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  }

  // forward-shift
  if (z > 0) {
    move_point = Point(sub_x, y, z-1, t, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = point.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
      dst = move_point.getPointGauge(static_cast<Complex*>(front_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  } else {
    // backward boundary to 0 line
    move_point = Point(sub_x, y, Lz-1, t, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = static_cast<Complex*>(front_boundary) + i * single_gauge + ((t * Ly + y) * Lx + x) * Nc * Nc;
      dst = move_point.getPointGauge(static_cast<Complex*>(front_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  }
}


__global__ void shiftGaugeT(void* origin_gauge, void* front_shift_gauge, void* back_shift_gauge, void* front_boundary, void* back_boundary, int Lx, int Ly, int Lz, int Lt) {
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Ly * Lx);
  int z = thread % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;

  int sub_x = (x >> 1);
  int single_gauge = Lx * Ly * Lz * Nc * Nc;
  int parity = (x + y + z + t) & 0x01;

  Complex* src;
  Complex* dst;

  Point point(sub_x, y, z, t, parity);
  Point move_point;

  // backward-shift
  // if (t > 0) {
  if (t < Lt - 1) {
    // move_point = Point(sub_x, y, z, t-1, 1-parity);
    move_point = Point(sub_x, y, z, t+1, 1-parity);
    for (int i = 0; i < Nd; i++) {  // i means four dims
      src = point.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
      dst = move_point.getPointGauge(static_cast<Complex*>(back_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  } else {
    // forward boundary to Lt-1 line
    // move_point = Point(sub_x, y, z, Lt-1, 1-parity);
    move_point = Point(sub_x, y, z, 0, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = static_cast<Complex*>(back_boundary) + i * single_gauge + ((z * Ly + y) * Lx + x) * Nc * Nc;
      // dst = move_point.getPointGauge(static_cast<Complex*>(back_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      dst = move_point.getPointGauge(static_cast<Complex*>(back_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  }

  // forward-shift
  // if (t < Lt-1) {
  if (t > 0) {
    move_point = Point(sub_x, y, z, t-1, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = point.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
      dst = move_point.getPointGauge(static_cast<Complex*>(front_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  } else {
    // backward boundary to 0 line
    move_point = Point(sub_x, y, z, Lt-1, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = static_cast<Complex*>(front_boundary) + i * single_gauge + ((z * Ly + y) * Lx + x) * Nc * Nc;
      dst = move_point.getPointGauge(static_cast<Complex*>(front_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  }
}

struct Coord {  // use this to store the coord of this process, and calculate adjacent process
  int x;
  int y;
  int z;
  int t;
  Coord() = default;
  Coord(int p_x, int p_y, int p_z, int p_t) : x(p_x), y(p_y), z(p_z), t(p_t) {}
  int calculateMpiRank() const {
    return ((x*grid_y + y)*grid_z+z)*grid_t + t;
  }
  Coord adjCoord(int front_back, int direction) const {
    // suppose all of grid_x, grid_y, grid_z, grid_t >= 1
    assert(front_back==FRONT || front_back==BACK);
    assert(direction==X_DIRECTION || direction==Y_DIRECTION || direction==Z_DIRECTION || direction==T_DIRECTION);

    int new_pos;
    switch (direction) {
      case X_DIRECTION:
        new_pos = (front_back == FRONT) ? ((x+1)%grid_x) : ((x+grid_x-1)%grid_x);
        return Coord(new_pos, y, z, t);
        break;
      case Y_DIRECTION:
        new_pos = (front_back == FRONT) ? ((y+1)%grid_y) : ((y+grid_y-1)%grid_y);
        return Coord(x, new_pos, z, t);
        break;
      case Z_DIRECTION:
        new_pos = (front_back == FRONT) ? ((z+1)%grid_z) : ((z+grid_z-1)%grid_z);
        return Coord(x, y, new_pos, t);
        break;
      case T_DIRECTION:
        new_pos = (front_back == FRONT) ? ((t+1)%grid_t) : ((t+grid_t-1)%grid_t);
        return Coord(x, y, z, new_pos);
        break;

      default:
        break;
    }
    return *this;
  }

  Coord& operator=(const Coord& rhs) {
    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
    t = rhs.t;
    return *this;
  }
};
static Coord coord;
MPICommunicator *mpi_comm;

void MPICommunicator::preDslash(void* fermion_in, int parity, int invert_flag) {
  for (int i = 0; i < Nd; i++) {
    // calc Boundary and send Boundary
    prepareFrontBoundaryVector (fermion_in, i, parity, invert_flag);
  }
  for (int i = 0; i < Nd; i++) {
    // recv Boundary
    recvBoundaryVector(i);
  }
}

void MPICommunicator::postDslash(void* fermion_out, int parity, int invert_flag) {
  assert (invert_flag == 0 || invert_flag == 1);
  Complex h_flag;
  Complex* d_flag_ptr;
  checkCudaErrors(cudaMalloc(&d_flag_ptr, sizeof(Complex)));

  if (invert_flag == 0) {
    h_flag = Complex(1, 0);
  } else {
    h_flag = Complex(-1, 0);
  }
  checkCudaErrors(cudaMemcpy(d_flag_ptr, &h_flag, sizeof(Complex), cudaMemcpyHostToDevice));

  Complex* h_addr;
  Complex* d_addr;
  int boundary_length;

  // Barrier
  for (int i = 0; i < Nd; i++) {
    int length[Nd] = {Lx_, Ly_, Lz_, Lt_};
    length[i] = 1;
    int sub_vol = 1;
    for (int j = 0; j < Nd; j++) {
      sub_vol *= length[j];
    }
    sub_vol >>= 1;  // div 2
    boundary_length = sub_vol * (Ns * Nc);

    dim3 subGridDim(sub_vol / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE);

    h_addr = mpi_comm->getHostRecvBufferAddr(FRONT, i);
    d_addr = mpi_comm->getRecvBufferAddr(FRONT, i);
    // src_process = grid_front[i];
    //calculateFrontBoundaryX(void* gauge, void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer)
    void *args1[] = {&gauge_, &fermion_out, &Lx_, &Ly_, &Lz_, &Lt_, &parity, &d_addr, &d_flag_ptr};
    if (i == T_DIRECTION && grid_t > 1) {
      // recv from front
      MPI_Wait(&recv_front_req[i], &recv_front_status[i]);
      checkCudaErrors(cudaMemcpy(d_addr, h_addr, sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaLaunchKernel((void *)calculateFrontBoundaryT, subGridDim, blockDim, args1));
      checkCudaErrors(cudaDeviceSynchronize());
    } else if (i == Z_DIRECTION && grid_z > 1) {
      // recv from front
      MPI_Wait(&recv_front_req[i], &recv_front_status[i]);
      checkCudaErrors(cudaMemcpy(d_addr, h_addr, sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaLaunchKernel((void *)calculateFrontBoundaryZ, subGridDim, blockDim, args1));
      checkCudaErrors(cudaDeviceSynchronize());
    } else if (i == Y_DIRECTION && grid_y > 1) {
      // recv from front
      MPI_Wait(&recv_front_req[i], &recv_front_status[i]);
      checkCudaErrors(cudaMemcpy(d_addr, h_addr, sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaLaunchKernel((void *)calculateFrontBoundaryY, subGridDim, blockDim, args1));
      checkCudaErrors(cudaDeviceSynchronize());
    } else if (i == X_DIRECTION && grid_x > 1) {
      // recv from front
      MPI_Wait(&recv_front_req[i], &recv_front_status[i]);
      checkCudaErrors(cudaMemcpy(d_addr, h_addr, sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaLaunchKernel((void *)calculateFrontBoundaryX, subGridDim, blockDim, args1));
      checkCudaErrors(cudaDeviceSynchronize());
    }

    h_addr = mpi_comm->getHostRecvBufferAddr(BACK, i);
    d_addr = mpi_comm->getRecvBufferAddr(BACK, i);
    // src_process = grid_back[i];
    // recv from front
    // calculateBackBoundaryT(void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer)
    void *args2[] = {&fermion_out, &Lx_, &Ly_, &Lz_, &Lt_, &parity, &d_addr};
    if (i == T_DIRECTION && grid_t > 1) {
      MPI_Wait(&recv_back_req[i], &recv_back_status[i]);
      checkCudaErrors(cudaMemcpy(d_addr, h_addr, sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaLaunchKernel((void *)calculateBackBoundaryT, subGridDim, blockDim, args2));
      checkCudaErrors(cudaDeviceSynchronize());
    } else if (i == Z_DIRECTION && grid_z > 1) {
      MPI_Wait(&recv_back_req[i], &recv_back_status[i]);
      checkCudaErrors(cudaMemcpy(d_addr, h_addr, sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaLaunchKernel((void *)calculateBackBoundaryZ, subGridDim, blockDim, args2));
      checkCudaErrors(cudaDeviceSynchronize());
    } else if (i == Y_DIRECTION && grid_y > 1) {
      MPI_Wait(&recv_back_req[i], &recv_back_status[i]);
      checkCudaErrors(cudaMemcpy(d_addr, h_addr, sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaLaunchKernel((void *)calculateBackBoundaryY, subGridDim, blockDim, args2));
      checkCudaErrors(cudaDeviceSynchronize());
    } else if (i == X_DIRECTION && grid_x > 1) {
      MPI_Wait(&recv_back_req[i], &recv_back_status[i]);
      checkCudaErrors(cudaMemcpy(d_addr, h_addr, sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaLaunchKernel((void *)calculateBackBoundaryX, subGridDim, blockDim, args2));
      checkCudaErrors(cudaDeviceSynchronize());
    }

    // send to front
    if ((i == T_DIRECTION && grid_t > 1) || 
      (i == Z_DIRECTION && grid_z > 1) || 
      (i == Y_DIRECTION && grid_y > 1) || 
      (i == X_DIRECTION && grid_x > 1) ) {
      MPI_Wait(&send_front_req[i], &send_front_status[i]);
      // send to back
      MPI_Wait(&send_back_req[i], &send_back_status[i]);
    }
  }
  // calc result
  checkCudaErrors(cudaFree(d_flag_ptr));
}

void MPICommunicator::recvBoundaryVector(int direction) {
  Complex* h_addr;
  int src_process;

  int length[Nd] = {Lx_, Ly_, Lz_, Lt_};
  length[direction] = 1;
  int boundary_length = 1;
  int sub_vol = 1;
  for (int i = 0; i < Nd; i++) {
    sub_vol *= length[i];
  }
  sub_vol >>= 1;  // div 2
  boundary_length = sub_vol * (Ns * Nc);

  // from front
  h_addr = mpi_comm->getHostRecvBufferAddr(FRONT, direction);
  src_process = grid_front[direction];
  if ((direction == T_DIRECTION && grid_t > 1) || 
      (direction == Z_DIRECTION && grid_z > 1) || 
      (direction == Y_DIRECTION && grid_y > 1) || 
      (direction == X_DIRECTION && grid_x > 1) )    {
    MPI_Irecv(h_addr, boundary_length * 2, MPI_DOUBLE, src_process, BACK, MPI_COMM_WORLD, &recv_front_req[direction]); // src_process tag is BACK, so use same tag, which is BACK(though from FRONT, so sad)
  }
  // from back
  h_addr = mpi_comm->getHostRecvBufferAddr(BACK, direction);
  src_process = grid_back[direction];
  if ((direction == T_DIRECTION && grid_t > 1) || 
      (direction == Z_DIRECTION && grid_z > 1) || 
      (direction == Y_DIRECTION && grid_y > 1) || 
      (direction == X_DIRECTION && grid_x > 1) )    {
    MPI_Irecv(h_addr, boundary_length * 2, MPI_DOUBLE, src_process, FRONT, MPI_COMM_WORLD, &recv_back_req[direction]);// src_process tag is FRONT, so use same tag, which is FRONT (though from FRONT, so sad)
  }
}

void MPICommunicator::prepareFrontBoundaryVector(void* fermion_in, int direction, int parity, int invert_flag) { // add parameter  invert_flag,  0---->flag(1,0)  1--->flag(-1, 0)
  assert (invert_flag == 0 || invert_flag == 1);
  Complex h_flag;
  Complex* d_flag_ptr;
  checkCudaErrors(cudaMalloc(&d_flag_ptr, sizeof(Complex)));

  if (invert_flag == 0) {
    h_flag = Complex(1, 0);
  } else {
    h_flag = Complex(-1, 0);
  }
  checkCudaErrors(cudaMemcpy(d_flag_ptr, &h_flag, sizeof(Complex), cudaMemcpyHostToDevice));


  Complex* h_addr;
  Complex* d_addr;

  int dst_process;
  int length[Nd] = {Lx_, Ly_, Lz_, Lt_};
  length[direction] = 1;
  int boundary_length = 1;
  int sub_vol = 1;
  for (int i = 0; i < Nd; i++) {
    sub_vol *= length[i];
  }
  sub_vol >>= 1;  // div 2
  boundary_length = sub_vol * (Ns * Nc);
  dim3 subGridDim(sub_vol / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);

  // front
  dst_process = grid_front[direction];
  h_addr = mpi_comm->getHostSendBufferAddr(FRONT, direction);
  d_addr = mpi_comm->getSendBufferAddr(FRONT, direction);

  void *args1[] = {&gauge_, &fermion_in, &Lx_, &Ly_, &Lz_, &Lt_, &parity, &d_addr, &d_flag_ptr};
  if (direction == T_DIRECTION && grid_t > 1) {
    checkCudaErrors(cudaLaunchKernel((void *)DslashTransferFrontT, subGridDim, blockDim, args1));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_addr, d_addr, sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
    MPI_Isend(h_addr, boundary_length * 2, MPI_DOUBLE, dst_process, FRONT, MPI_COMM_WORLD, &send_front_req[direction]);
  } else if (direction == Z_DIRECTION && grid_z > 1) {
    checkCudaErrors(cudaLaunchKernel((void *)DslashTransferFrontZ, subGridDim, blockDim, args1));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_addr, d_addr, sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
    MPI_Isend(h_addr, boundary_length * 2, MPI_DOUBLE, dst_process, FRONT, MPI_COMM_WORLD, &send_front_req[direction]);
  } else if (direction == Y_DIRECTION && grid_y > 1) {
    checkCudaErrors(cudaLaunchKernel((void *)DslashTransferFrontY, subGridDim, blockDim, args1));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_addr, d_addr, sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
    MPI_Isend(h_addr, boundary_length * 2, MPI_DOUBLE, dst_process, FRONT, MPI_COMM_WORLD, &send_front_req[direction]);
  } else if (direction == X_DIRECTION && grid_x > 1) {
    checkCudaErrors(cudaLaunchKernel((void *)DslashTransferFrontX, subGridDim, blockDim, args1));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_addr, d_addr, sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
    MPI_Isend(h_addr, boundary_length * 2, MPI_DOUBLE, dst_process, FRONT, MPI_COMM_WORLD, &send_front_req[direction]);
  }


  // back
  dst_process = grid_back[direction];
  h_addr = mpi_comm->getHostSendBufferAddr(BACK, direction);
  d_addr = mpi_comm->getSendBufferAddr(BACK, direction);
  void *args2[] = {&fermion_in, &Lx_, &Ly_, &Lz_, &Lt_, &parity, &d_addr};
  if (direction == T_DIRECTION && grid_t > 1) {
    checkCudaErrors(cudaLaunchKernel((void *)DslashTransferBackT, subGridDim, blockDim, args2));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_addr, d_addr, sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
    MPI_Isend(h_addr, boundary_length * 2, MPI_DOUBLE, dst_process, BACK, MPI_COMM_WORLD, &send_back_req[direction]);
  } else if (direction == Z_DIRECTION && grid_z > 1) {
    checkCudaErrors(cudaLaunchKernel((void *)DslashTransferBackZ, subGridDim, blockDim, args2));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_addr, d_addr, sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
    MPI_Isend(h_addr, boundary_length * 2, MPI_DOUBLE, dst_process, BACK, MPI_COMM_WORLD, &send_back_req[direction]);
  } else if (direction == Y_DIRECTION && grid_y > 1) {
    checkCudaErrors(cudaLaunchKernel((void *)DslashTransferBackY, subGridDim, blockDim, args2));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_addr, d_addr, sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
    MPI_Isend(h_addr, boundary_length * 2, MPI_DOUBLE, dst_process, BACK, MPI_COMM_WORLD, &send_back_req[direction]);
  } else if (direction == X_DIRECTION && grid_x > 1) {
    checkCudaErrors(cudaLaunchKernel((void *)DslashTransferBackX, subGridDim, blockDim, args2));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_addr, d_addr, sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
    MPI_Isend(h_addr, boundary_length * 2, MPI_DOUBLE, dst_process, BACK, MPI_COMM_WORLD, &send_back_req[direction]);
  }
  checkCudaErrors(cudaFree(d_flag_ptr));
}

void MPICommunicator::prepareGauge() {

  for (int i = 0; i < Nd; i++) {
    shiftGauge(gauge_, gauge_shift[i][FRONT], gauge_shift[i][BACK], i);
  }

  // twice shift
  // xy-0    xz-1    xt-2
  // yz-3    yt-4    zt-5
  // 0 means ++     1 means +-     2 means -+     3 means --
  shiftGauge(gauge_shift[X_DIRECTION][FRONT], gauge_twice_shift[0][0], gauge_twice_shift[0][1], Y_DIRECTION);
  shiftGauge(gauge_shift[X_DIRECTION][BACK], gauge_twice_shift[0][2], gauge_twice_shift[0][3], Y_DIRECTION);

  shiftGauge(gauge_shift[X_DIRECTION][FRONT], gauge_twice_shift[1][0], gauge_twice_shift[1][1], Z_DIRECTION);
  shiftGauge(gauge_shift[X_DIRECTION][BACK], gauge_twice_shift[1][2], gauge_twice_shift[1][3], Z_DIRECTION);

  shiftGauge(gauge_shift[X_DIRECTION][FRONT], gauge_twice_shift[2][0], gauge_twice_shift[2][1], T_DIRECTION);
  shiftGauge(gauge_shift[X_DIRECTION][BACK], gauge_twice_shift[2][2], gauge_twice_shift[2][3], T_DIRECTION);

  shiftGauge(gauge_shift[Y_DIRECTION][FRONT], gauge_twice_shift[3][0], gauge_twice_shift[3][1], Z_DIRECTION);
  shiftGauge(gauge_shift[Y_DIRECTION][BACK], gauge_twice_shift[3][2], gauge_twice_shift[3][3], Z_DIRECTION);

  shiftGauge(gauge_shift[Y_DIRECTION][FRONT], gauge_twice_shift[4][0], gauge_twice_shift[4][1], T_DIRECTION);
  shiftGauge(gauge_shift[Y_DIRECTION][BACK], gauge_twice_shift[4][2], gauge_twice_shift[4][3], T_DIRECTION);

  shiftGauge(gauge_shift[Z_DIRECTION][FRONT], gauge_twice_shift[5][0], gauge_twice_shift[5][1], T_DIRECTION);
  shiftGauge(gauge_shift[Z_DIRECTION][BACK], gauge_twice_shift[5][2], gauge_twice_shift[5][3], T_DIRECTION);
}

void MPICommunicator::shiftGauge(void* src_gauge, void* front_shift_gauge, void* back_shift_gauge, int direction) {
  prepareBoundaryGauge(src_gauge, direction);      // 2nd and 3rd parameter ----send buffer
  sendGauge(direction);
  recvGauge(direction);
  qcuGaugeMPIBarrier(direction);
  shiftGaugeKernel(src_gauge, front_shift_gauge, back_shift_gauge, direction);
}

void MPICommunicator::shiftGaugeKernel(void* src_gauge, void* front_shift_gauge, void* back_shift_gauge, int direction) {
  int vol = Lx_ * Ly_ * Lz_ * Lt_;
  dim3 gridDim(vol / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);

  void *args[] = {&src_gauge, &front_shift_gauge, &back_shift_gauge, &d_recv_gauge[direction][FRONT], &d_recv_gauge[direction][BACK], &Lx_, &Ly_, &Lz_, &Lt_};
  switch(direction) {
    case X_DIRECTION:
      checkCudaErrors(cudaLaunchKernel((void *)shiftGaugeX, gridDim, blockDim, args));
      break;
    case Y_DIRECTION:
      checkCudaErrors(cudaLaunchKernel((void *)shiftGaugeY, gridDim, blockDim, args));
      break;
    case Z_DIRECTION:
      checkCudaErrors(cudaLaunchKernel((void *)shiftGaugeZ, gridDim, blockDim, args));
      break;
    case T_DIRECTION:
      checkCudaErrors(cudaLaunchKernel((void *)shiftGaugeT, gridDim, blockDim, args));
      break;
    default:
      break;
  }
  checkCudaErrors(cudaDeviceSynchronize());
}

void MPICommunicator::prepareBoundaryGauge(void* src_gauge, int direction) {
  void *args[] = {&src_gauge, &d_send_gauge[direction][FRONT], &d_send_gauge[direction][BACK], &Lx_, &Ly_, &Lz_, &Lt_};
  int length[Nd] = {Lx_, Ly_, Lz_, Lt_};
  length[direction] = 1;
  int full_length = 1;
  for (int i = 0; i < Nd; i++) {
    full_length *= length[i];
  }

  dim3 gridDim(full_length / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  // sendGaugeBoundaryToBufferX(void* origin_gauge, void* front_buffer, void* back_buffer, int Lx, int Ly, int Lz, int Lt)
  switch (direction) {
    case X_DIRECTION:
      checkCudaErrors(cudaLaunchKernel((void *)sendGaugeBoundaryToBufferX, gridDim, blockDim, args));
      break;
    case Y_DIRECTION:
      checkCudaErrors(cudaLaunchKernel((void *)sendGaugeBoundaryToBufferY, gridDim, blockDim, args));
      break;
    case Z_DIRECTION:
      checkCudaErrors(cudaLaunchKernel((void *)sendGaugeBoundaryToBufferZ, gridDim, blockDim, args));
      break;
    case T_DIRECTION:
      checkCudaErrors(cudaLaunchKernel((void *)sendGaugeBoundaryToBufferT, gridDim, blockDim, args));
      break;
    default:
      break;
  }
  checkCudaErrors(cudaDeviceSynchronize());
}

void MPICommunicator::qcuGaugeMPIBarrier(int direction) {
  int process;
  int length[Nd] = {Lx_, Ly_, Lz_, Lt_};
  length[direction] = 1;
  int boundary_length = 1;
  for (int i = 0; i < Nd; i++) {
    boundary_length *= length[i];
  }
  boundary_length *= (Nd * Nc * Nc);
  // from front process
  process = grid_front[direction];
  if (process_rank != process) {
    // recv
    MPI_Wait(&recv_front_req[direction], &recv_front_status[direction]);
    // send
    MPI_Wait(&send_front_req[direction], &send_front_status[direction]);
    checkCudaErrors(cudaMemcpy(d_recv_gauge[direction][FRONT], h_recv_gauge[direction][FRONT], sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
  } else {
    checkCudaErrors(cudaMemcpy(d_recv_gauge[direction][FRONT], d_send_gauge[direction][BACK], sizeof(Complex) * boundary_length, cudaMemcpyDeviceToDevice));
  }
  // from back process
  process = grid_back[direction];
  if (process_rank != process) {
    // recv
    MPI_Wait(&recv_back_req[direction], &recv_back_status[direction]);
    // send
    MPI_Wait(&send_back_req[direction], &send_back_status[direction]);
    checkCudaErrors(cudaMemcpy(d_recv_gauge[direction][BACK], h_recv_gauge[direction][BACK], sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
  } else {
    checkCudaErrors(cudaMemcpy(d_recv_gauge[direction][BACK], d_send_gauge[direction][FRONT], sizeof(Complex) * boundary_length, cudaMemcpyDeviceToDevice));
  }
}

void MPICommunicator::recvGauge(int direction) {
  int src_process;
  int length[Nd] = {Lx_, Ly_, Lz_, Lt_};
  length[direction] = 1;
  int boundary_length = 1;
  for (int i = 0; i < Nd; i++) {
    boundary_length *= length[i];
  }

  boundary_length *= (Nd * Nc * Nc);
  src_process = grid_front[direction];
  if (process_rank != src_process) {  // send buffer to other process
    MPI_Irecv(h_recv_gauge[direction][FRONT], boundary_length*2, MPI_DOUBLE, src_process, BACK, MPI_COMM_WORLD, &recv_front_req[direction]);
  }
  // back
  src_process = grid_back[direction];
  if (process_rank != src_process) {  // send buffer to other process
    MPI_Irecv(h_recv_gauge[direction][BACK], boundary_length*2, MPI_DOUBLE, src_process, FRONT, MPI_COMM_WORLD, &recv_back_req[direction]);
  }
}

void MPICommunicator::sendGauge(int direction) {
  int dst_process;
  int length[Nd] = {Lx_, Ly_, Lz_, Lt_};
  length[direction] = 1;
  int boundary_length = 1;
  for (int i = 0; i < Nd; i++) {
    boundary_length *= length[i];
  }
  boundary_length *= (Nd * Nc * Nc);
  // front
  dst_process = grid_front[direction];
  if (process_rank != dst_process) {  // send buffer to other process
    checkCudaErrors(cudaMemcpy(h_send_gauge[direction][FRONT], d_send_gauge[direction][FRONT], sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
    MPI_Isend(h_send_gauge[direction][FRONT], boundary_length*2, MPI_DOUBLE, dst_process, FRONT, MPI_COMM_WORLD, &send_front_req[direction]);
    }
  // back
  dst_process = grid_back[direction];
  if (process_rank != dst_process) {  // send buffer to other process
    checkCudaErrors(cudaMemcpy(h_send_gauge[direction][BACK], d_send_gauge[direction][BACK], sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
    MPI_Isend(h_send_gauge[direction][BACK], boundary_length*2, MPI_DOUBLE, dst_process, BACK, MPI_COMM_WORLD, &send_back_req[direction]);
  }
}

void MPICommunicator::allocateGaugeBuffer() {
  int total_vol = Nd * Lt_ * Lz_ * Ly_ * Lx_ * Nc * Nc;
  // TODO: allocate  gauge 
  for (int i = 0; i < Nd; i++) {
    for (int j = 0; j < 2; j++) {
      checkCudaErrors(cudaMalloc(&gauge_shift[i][j], sizeof(Complex) * total_vol));
    }
  }
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 4; j++) {
      checkCudaErrors(cudaMalloc(&gauge_twice_shift[i][j], sizeof(Complex) * total_vol));
    }
  }

  // TODO: allocate boundary gauge
  int boundary_size;
  // int length[4] = {Lx_, Ly_, Lz_, Lt_};
  for (int i = 0; i < Nd; i++) {
    // int shift[Nd] = {1, 1, 1, 1};
    int length[4] = {Lx_, Ly_, Lz_, Lt_};
    length[i] = 1;
    boundary_size = 1;
    for (int j = 0; j < Nd; j++) {
      boundary_size *= length[j];
    }
    boundary_size *= (Nd * Nc * Nc);
    for (int j = 0; j < 2; j++) {
      h_send_gauge[i][j] = new Complex[boundary_size];
      h_recv_gauge[i][j] = new Complex[boundary_size];
      checkCudaErrors(cudaMalloc(&d_send_gauge[i][j], sizeof(Complex) * boundary_size));
      checkCudaErrors(cudaMalloc(&d_recv_gauge[i][j], sizeof(Complex) * boundary_size));
    }
  }
}
void MPICommunicator::allocateBuffer() {

  checkCudaErrors(cudaMalloc(&d_partial_result_buffer, sizeof(Complex) * Lx_ * Ly_ * Lz_ * Lt_ / BLOCK_SIZE));
  int boundary_size;
  if (grid_x != 1) {
    boundary_size = Ly_ * Lz_ * Lt_ * Ns * Nc / 2;
    h_send_front_vec[0] = new Complex[boundary_size];
    h_send_back_vec[0] = new Complex[boundary_size];
    h_recv_front_vec[0] = new Complex[boundary_size];
    h_recv_back_vec[0] = new Complex[boundary_size];
    checkCudaErrors(cudaMalloc(&d_send_front_vec[0], sizeof(Complex) * boundary_size));
    checkCudaErrors(cudaMalloc(&d_send_back_vec[0], sizeof(Complex) * boundary_size));
    checkCudaErrors(cudaMalloc(&d_recv_front_vec[0], sizeof(Complex) * boundary_size));
    checkCudaErrors(cudaMalloc(&d_recv_back_vec[0], sizeof(Complex) * boundary_size));
  }
  if (grid_y != 1) {
    boundary_size = Lx_ * Lz_ * Lt_ * Ns * Nc / 2;
    h_send_front_vec[1] = new Complex[boundary_size];
    h_send_back_vec[1] = new Complex[boundary_size];
    h_recv_front_vec[1] = new Complex[boundary_size];
    h_recv_back_vec[1] = new Complex[boundary_size];
    checkCudaErrors(cudaMalloc(&d_send_front_vec[1], sizeof(Complex) * boundary_size));
    checkCudaErrors(cudaMalloc(&d_send_back_vec[1], sizeof(Complex) * boundary_size));
    checkCudaErrors(cudaMalloc(&d_recv_front_vec[1], sizeof(Complex) * boundary_size));
    checkCudaErrors(cudaMalloc(&d_recv_back_vec[1], sizeof(Complex) * boundary_size));
  }
  if (grid_z != 1) {
    boundary_size = Lx_ * Ly_ * Lt_ * Ns * Nc / 2;
    h_send_front_vec[2] = new Complex[boundary_size];
    h_send_back_vec[2] = new Complex[boundary_size];
    h_recv_front_vec[2] = new Complex[boundary_size];
    h_recv_back_vec[2] = new Complex[boundary_size];
    checkCudaErrors(cudaMalloc(&d_send_front_vec[2], sizeof(Complex) * boundary_size));
    checkCudaErrors(cudaMalloc(&d_send_back_vec[2], sizeof(Complex) * boundary_size));
    checkCudaErrors(cudaMalloc(&d_recv_front_vec[2], sizeof(Complex) * boundary_size));
    checkCudaErrors(cudaMalloc(&d_recv_back_vec[2], sizeof(Complex) * boundary_size));
  }
  if (grid_t != 1) {
    boundary_size = Lx_ * Ly_ * Lz_ * Ns * Nc / 2;
    h_send_front_vec[3] = new Complex[boundary_size];
    h_send_back_vec[3] = new Complex[boundary_size];
    h_recv_front_vec[3] = new Complex[boundary_size];
    h_recv_back_vec[3] = new Complex[boundary_size];
    checkCudaErrors(cudaMalloc(&(d_send_front_vec[3]), sizeof(Complex) * boundary_size));
    checkCudaErrors(cudaMalloc(&(d_send_back_vec[3]), sizeof(Complex) * boundary_size));
    checkCudaErrors(cudaMalloc(&(d_recv_front_vec[3]), sizeof(Complex) * boundary_size));
    checkCudaErrors(cudaMalloc(&(d_recv_back_vec[3]), sizeof(Complex) * boundary_size));
  }
}
void MPICommunicator::calculateAdjacentProcess() {
  grid_front[0] = coord.adjCoord(FRONT, X_DIRECTION).calculateMpiRank();
  grid_back[0] = coord.adjCoord(BACK, X_DIRECTION).calculateMpiRank();
  grid_front[1] = coord.adjCoord(FRONT, Y_DIRECTION).calculateMpiRank();
  grid_back[1] = coord.adjCoord(BACK, Y_DIRECTION).calculateMpiRank();
  grid_front[2] = coord.adjCoord(FRONT, Z_DIRECTION).calculateMpiRank();
  grid_back[2] = coord.adjCoord(BACK, Z_DIRECTION).calculateMpiRank();
  grid_front[3] = coord.adjCoord(FRONT, T_DIRECTION).calculateMpiRank();
  grid_back[3] = coord.adjCoord(BACK, T_DIRECTION).calculateMpiRank();
}

void MPICommunicator::freeBuffer() {
  checkCudaErrors(cudaFree(d_partial_result_buffer));
  if (grid_x != 1) {
    delete h_send_front_vec[0];
    delete h_send_back_vec[0];
    delete h_recv_front_vec[0];
    delete h_recv_back_vec[0];
    checkCudaErrors(cudaFree(d_send_front_vec[0]));
    checkCudaErrors(cudaFree(d_send_back_vec[0]));
    checkCudaErrors(cudaFree(d_recv_front_vec[0]));
    checkCudaErrors(cudaFree(d_recv_back_vec[0]));
  }
  if (grid_y != 1) {
    delete h_send_front_vec[1];
    delete h_send_back_vec[1];
    delete h_recv_front_vec[1];
    delete h_recv_back_vec[1];
    checkCudaErrors(cudaFree(d_send_front_vec[1]));
    checkCudaErrors(cudaFree(d_send_back_vec[1]));
    checkCudaErrors(cudaFree(d_recv_front_vec[1]));
    checkCudaErrors(cudaFree(d_recv_back_vec[1]));
  }
  if (grid_z != 1) {
    delete h_send_front_vec[2];
    delete h_send_back_vec[2];
    delete h_recv_front_vec[2];
    delete h_recv_back_vec[2];
    checkCudaErrors(cudaFree(d_send_front_vec[2]));
    checkCudaErrors(cudaFree(d_send_back_vec[2]));
    checkCudaErrors(cudaFree(d_recv_front_vec[2]));
    checkCudaErrors(cudaFree(d_recv_back_vec[2]));
  }
  if (grid_t != 1) {
    delete h_send_front_vec[3];
    delete h_send_back_vec[3];
    delete h_recv_front_vec[3];
    delete h_recv_back_vec[3];
    checkCudaErrors(cudaFree(d_send_front_vec[3]));
    checkCudaErrors(cudaFree(d_send_back_vec[3]));
    checkCudaErrors(cudaFree(d_recv_front_vec[3]));
    checkCudaErrors(cudaFree(d_recv_back_vec[3]));
  }
  printf("vector buffer free... over\n");
}
void MPICommunicator::freeGaugeBuffer () {
  for (int i = 0; i < Nd; i++) {
    for (int j = 0; j < 2; j++) {
      checkCudaErrors(cudaFree(gauge_shift[i][j]));
    }
  }
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 4; j++) {
      checkCudaErrors(cudaFree(gauge_twice_shift[i][j]));
    }
  }

  for (int i = 0; i < Nd; i++) {
    for (int j = 0; j < 2; j++) {
      checkCudaErrors(cudaFree(d_send_gauge[i][j]));
      checkCudaErrors(cudaFree(d_recv_gauge[i][j]));
      delete h_send_gauge[i][j];
      delete h_recv_gauge[i][j];
    }
  }
  printf("Gauge buffer free... over\n");
}


// parameters on device!!!   ax+y ----> y
void MPICommunicator::interprocess_saxpy_barrier(void* x, void* y, void* scalar, int vol) {
  // int vol = Lx * Ly * Lz * Lt;
  gpu_saxpy(x, y, scalar, vol);
  MPI_Barrier(MPI_COMM_WORLD);
}

void MPICommunicator::interprocess_sax_barrier (void* x, void* scalar, int vol) {
  gpu_sclar_multiply_vector(x, scalar, vol);
  MPI_Barrier(MPI_COMM_WORLD);
}

void MPICommunicator::interprocess_inner_prod_barrier(void* x, void* y, void* result, int vol) {
  Complex all_reduce_result;
  Complex single_process_result;
  // void gpu_inner_product (void* x, void* y, void* result, void* partial_result, int vol) 
  gpu_inner_product(x, y, result, d_partial_result_buffer, vol);

  checkCudaErrors(cudaMemcpy(&single_process_result, result, sizeof(Complex), cudaMemcpyDeviceToHost));
  MPI_Allreduce(&single_process_result, &all_reduce_result, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  checkCudaErrors(cudaMemcpy(result, &all_reduce_result, sizeof(Complex), cudaMemcpyHostToDevice));
}

void initMPICommunicator(void* gauge, void* fermion_in, void* fermion_out, int Lx, int Ly, int Lz, int Lt) {
  mpi_comm = new MPICommunicator(gauge, fermion_in, fermion_out, Lx, Ly, Lz, Lt);
}

// initialize the struct of grid, first function to call
void initGridSize(QcuGrid_t* grid, QcuParam* p_param, void* gauge, void* fermion_in, void* fermion_out) {
  // x,y,z,t
  int Lx = p_param->lattice_size[0];
  int Ly = p_param->lattice_size[1];
  int Lz = p_param->lattice_size[2];
  int Lt = p_param->lattice_size[3];

  grid_x = grid->grid_size[0];
  grid_y = grid->grid_size[1];
  grid_z = grid->grid_size[2];
  grid_t = grid->grid_size[3];

  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &process_num);
  coord.t = process_rank % grid_t;
  coord.z = process_rank / grid_t % grid_z;
  coord.y = process_rank / grid_t / grid_z % grid_y;
  coord.x = process_rank / grid_t / grid_z / grid_y;  // rank divide(g_y*g_z*g_t)

  initMPICommunicator(gauge, fermion_in, fermion_out, Lx, Ly, Lz, Lt);
}


__attribute__((constructor)) void initialize_mpi() {
  mpi_comm = nullptr;
  printf(RED"constructor\n");
  printf(CLR"");
}
__attribute__((destructor)) void destroySpace_mpi() {
  delete mpi_comm;
}