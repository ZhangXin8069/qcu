#include "qcu_wilson_dslash_neo.cuh"
#include "qcu_complex.cuh"
#include "qcu_point.cuh"
#include "qcu_communicator.cuh"
#include "qcu_shift_storage.cuh"
#include <chrono>


extern int grid_x;
extern int grid_y;
extern int grid_z;
extern int grid_t;
extern MPICommunicator *mpi_comm;

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



static __device__ inline void reconstructSU3(Complex *su3)
{
  su3[6] = (su3[1] * su3[5] - su3[2] * su3[4]).conj();
  su3[7] = (su3[2] * su3[3] - su3[0] * su3[5]).conj();
  su3[8] = (su3[0] * su3[4] - su3[1] * su3[3]).conj();
}

__device__ inline void loadGauge(Complex* u_local, void* gauge_ptr, int direction, const Point& p, int Lx, int Ly, int Lz, int Lt) {
  Complex* u = p.getPointGauge(static_cast<Complex*>(gauge_ptr), direction, Lx, Ly, Lz, Lt);
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    u_local[i] = u[i];
  }
  reconstructSU3(u_local);
}
__device__ inline void loadVector(Complex* src_local, void* fermion_in, const Point& p, int Lx, int Ly, int Lz, int Lt) {
  Complex* src = p.getPointVector(static_cast<Complex *>(fermion_in), Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src[i];
  }
}



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

  // x back   x==0 && parity == eo
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




void WilsonDslash::calculateDslash(int invert_flag) {
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
  // mpiDslashCoalesce(void *gauge, void *fermion_in, void *fermion_out,int Lx, int Ly, int Lz, int Lt, int parity, int grid_x, int grid_y, int grid_z, int grid_t, void* flag_ptr)
  // checkCudaErrors(cudaLaunchKernel((void *)mpiDslash, gridDim, blockDim, args));
  checkCudaErrors(cudaLaunchKernel((void *)mpiDslashCoalesce, gridDim, blockDim, args));

  checkCudaErrors(cudaDeviceSynchronize());
  // boundary calculate
  mpi_comm->postDslash(dslashParam_->fermion_out, parity, invert_flag);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  // printf("total time: (without malloc free memcpy) : %.9lf sec\n", double(duration) / 1e9);

  checkCudaErrors(cudaFree(d_flag));
}




static __device__ inline void loadGaugeCoalesced(Complex* u_local, void* gauge_ptr, int direction, const Point& p, int sub_Lx, int Ly, int Lz, int Lt, int thread_id) {
  double* start_ptr = p.getCoalescedGaugeAddr (gauge_ptr, direction, sub_Lx, Ly, Lz, Lt, thread_id);
  double real;
  double imag;
  int sub_vol = sub_Lx * Ly * Lz * Lt;
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    real = *start_ptr;
    start_ptr += sub_vol;
    imag = *start_ptr;
    start_ptr += sub_vol;
    u_local[i] = Complex(real, imag);
  }
  reconstructSU3(u_local);
}

static __device__ inline void loadVectorCoalesced(Complex* src_local, void* fermion_in, const Point& p, int half_Lx, int Ly, int Lz, int Lt, int thread_id) {
  double* start_ptr = p.getCoalescedVectorAddr (fermion_in, thread_id);
  double real;
  double imag;
  int sub_vol = sub_Lx * Ly * Lz * Lt;
  for (int i = 0; i < Ns * Nc; i++) {
    real = *start_ptr;
    start_ptr += sub_vol;
    imag = *start_ptr;
    start_ptr += sub_vol;
    src_local[i] = Complex(real, imag);
  }
}



static __global__ void mpiDslashCoalesce(void *gauge, void *fermion_in, void *fermion_out,int Lx, int Ly, int Lz, int Lt, int parity, int grid_x, int grid_y, int grid_z, int grid_t, void* flag_ptr) {
  assert(parity == 0 || parity == 1);

  Lx >>= 1;

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread_id / (Lz * Ly * Lx);
  int z = thread_id % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread_id % (Ly * Lx) / Lx;
  int x = thread_id % Lx;

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
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt, thread_id);
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

  // x back   x==0 && parity == eo
  move_point = p.move(BACK, 0, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, 0, move_point, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt, thread_id);

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
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt, thread_id);

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
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt, thread_id);


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
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt, thread_id);
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
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt, thread_id);

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
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt, thread_id);

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
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt, thread_id);

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

void shiftMemoryWilsonDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag) {
  void* coalesced_fermion_in;
  void* coalesced_fermion_out;
  void* coalesced_fermion_gauge;

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;

  checkCudaErrors(cudaMalloc(&coalesced_fermion_in, sizeof(double) * vol / 2 * Ns * Nc * 2));
  checkCudaErrors(cudaMalloc(&coalesced_fermion_out, sizeof(double) * vol / 2 * Ns * Nc * 2));
  checkCudaErrors(cudaMalloc(&coalesced_fermion_out, sizeof(double) * Nd * vol * (Nc-1) * Nc * 2));

  shiftVectorStorage(coalesced_fermion_in, fermion_in, TO_COALESCE, Lx, Ly, Lz, Lt);



  shiftVectorStorage(fermion_out, coalesced_fermion_out, TO_COALESCE, Lx, Ly, Lz, Lt);

  checkCudaErrors(cudaFree(coalesced_fermion_in));
  checkCudaErrors(cudaFree(coalesced_fermion_out));
  checkCudaErrors(cudaFree(coalesced_fermion_out));
}
void callWilsonDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag) {


  void* coalesced_fermion_in;
  void* coalesced_fermion_out;
  void* coalesced_fermion_gauge;

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;

  checkCudaErrors(cudaMalloc(&coalesced_fermion_in, sizeof(double) * vol / 2 * Ns * Nc * 2));
  checkCudaErrors(cudaMalloc(&coalesced_fermion_out, sizeof(double) * vol / 2 * Ns * Nc * 2));
  checkCudaErrors(cudaMalloc(&coalesced_fermion_out, sizeof(double) * Nd * vol * (Nc-1) * Nc * 2));

  shiftVectorStorage(coalesced_fermion_in, fermion_in, TO_COALESCE, Lx, Ly, Lz, Lt);


  DslashParam dslash_param(fermion_in, coalesced_fermion_in, gauge, param, parity);
  WilsonDslash dslash_solver(dslash_param);
  dslash_solver.calculateDslash();



  // shiftVectorStorage(fermion_out, coalesced_fermion_out, TO_COALESCE, Lx, Ly, Lz, Lt);

  checkCudaErrors(cudaFree(coalesced_fermion_in));
  checkCudaErrors(cudaFree(coalesced_fermion_out));
  checkCudaErrors(cudaFree(coalesced_fermion_out));



}
