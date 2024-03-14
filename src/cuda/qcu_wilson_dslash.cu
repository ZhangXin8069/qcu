#include "qcu_wilson_dslash.cuh"
#include "qcu_complex.cuh"
#include "qcu_point.cuh"
#include "qcu_communicator.cuh"
#include <chrono>

// #include "qcu_macro.cuh"

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



// // WARP version, no sync
// static __device__ void loadVectorBySharedMemory(void* shared_ptr, Complex* origin, Complex* result) {
//   // result is register variable
//   // __shared__ double shared_buffer[2 * Ns * Nc * BLOCK_SIZE];
//   double* shared_buffer = static_cast<double*>(shared_ptr);
//   int thread = blockDim.x * blockIdx.x + threadIdx.x;
//   int warp_index = (thread - thread / BLOCK_SIZE * BLOCK_SIZE) / WARP_SIZE;//thread % BLOCK_SIZE / WARP_SIZE;
//   Complex* shared_dst = reinterpret_cast<Complex*>(shared_buffer) + threadIdx.x * Ns * Nc;
//   Complex* warp_dst = origin + (thread / WARP_SIZE * WARP_SIZE) * Ns * Nc;
//   double* double_dst = reinterpret_cast<double*>(warp_dst);

//   // store result of shared memory to global memory
//   for (int i = threadIdx.x - threadIdx.x / WARP_SIZE * WARP_SIZE; i < WARP_SIZE * Ns * Nc * 2; i += WARP_SIZE) {
//     shared_buffer[warp_index * WARP_SIZE * Ns * Nc * 2 + i] = double_dst[i];
//   }
//   // store result to shared memory
//   for (int i = 0; i < Ns * Nc; i++) {
//     result[i] = shared_dst[i];
//   }
// }


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

// __global__ void gpuDslash(void *gauge, void *fermion_in, void *fermion_out,int Lx, int Ly, int Lz, int Lt, int parity)
// {
//   assert(parity == 0 || parity == 1);

//   __shared__ double shared_output_vec[BLOCK_SIZE * Ns * Nc * 2];
//   Lx >>= 1;

//   int thread = blockIdx.x * blockDim.x + threadIdx.x;
//   int t = thread / (Lz * Ly * Lx);
//   int z = thread % (Lz * Ly * Lx) / (Ly * Lx);
//   int y = thread % (Ly * Lx) / Lx;
//   int x = thread % Lx;

//   Complex u_local[Nc * Nc];   // for GPU
//   Complex src_local[Ns * Nc]; // for GPU
//   Complex dst_local[Ns * Nc]; // for GPU

//   Point p(x, y, z, t, parity);
//   Point move_point;


//   Complex temp;
//   for (int i = 0; i < Ns * Nc; i++) {
//     dst_local[i].clear2Zero();
//   }

//   // \mu = 1
//   loadGauge(u_local, gauge, 0, p, Lx, Ly, Lz, Lt);
//   move_point = p.move(FRONT, 0, Lx, Ly, Lz, Lt);
//   loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

//   for (int i = 0; i < Nc; i++) {
//     for (int j = 0; j < Nc; j++) {
//       // first row vector with col vector
//       temp = (src_local[0 * Nc + j] - src_local[3 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
//       dst_local[0 * Nc + i] += temp;
//       dst_local[3 * Nc + i] += temp * Complex(0, 1);
//       // second row vector with col vector
//       temp = (src_local[1 * Nc + j] - src_local[2 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
//       dst_local[1 * Nc + i] += temp;
//       dst_local[2 * Nc + i] += temp * Complex(0, 1);
//     }
//   }

//   move_point = p.move(BACK, 0, Lx, Ly, Lz, Lt);
//   loadGauge(u_local, gauge, 0, move_point, Lx, Ly, Lz, Lt);
//   loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

//   for (int i = 0; i < Nc; i++) {
//     for (int j = 0; j < Nc; j++) {
//       // first row vector with col vector
//       temp = (src_local[0 * Nc + j] + src_local[3 * Nc + j] * Complex(0, 1)) *
//              u_local[j * Nc + i].conj(); // transpose and conj
//       dst_local[0 * Nc + i] += temp;
//       dst_local[3 * Nc + i] += temp * Complex(0, -1);
//       // second row vector with col vector
//       temp = (src_local[1 * Nc + j] + src_local[2 * Nc + j] * Complex(0, 1)) *
//              u_local[j * Nc + i].conj(); // transpose and conj
//       dst_local[1 * Nc + i] += temp;
//       dst_local[2 * Nc + i] += temp * Complex(0, -1);
//     }
//   }

//   // \mu = 2
//   loadGauge(u_local, gauge, 1, p, Lx, Ly, Lz, Lt);
//   move_point = p.move(FRONT, 1, Lx, Ly, Lz, Lt);
//   loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

//   for (int i = 0; i < Nc; i++) {
//     for (int j = 0; j < Nc; j++) {
//       // first row vector with col vector
//       temp = (src_local[0 * Nc + j] + src_local[3 * Nc + j]) * u_local[i * Nc + j];
//       dst_local[0 * Nc + i] += temp;
//       dst_local[3 * Nc + i] += temp;
//       // second row vector with col vector
//       temp = (src_local[1 * Nc + j] - src_local[2 * Nc + j]) * u_local[i * Nc + j];
//       dst_local[1 * Nc + i] += temp;
//       dst_local[2 * Nc + i] += -temp;
//     }
//   }

//   move_point = p.move(BACK, 1, Lx, Ly, Lz, Lt);
//   loadGauge(u_local, gauge, 1, move_point, Lx, Ly, Lz, Lt);
//   loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

//   for (int i = 0; i < Nc; i++) {
//     for (int j = 0; j < Nc; j++) {
//       // first row vector with col vector
//       temp = (src_local[0 * Nc + j] - src_local[3 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
//       dst_local[0 * Nc + i] += temp;
//       dst_local[3 * Nc + i] += -temp;
//       // second row vector with col vector
//       temp = (src_local[1 * Nc + j] + src_local[2 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
//       dst_local[1 * Nc + i] += temp;
//       dst_local[2 * Nc + i] += temp;
//     }
//   }

//   // \mu = 3
//   loadGauge(u_local, gauge, 2, p, Lx, Ly, Lz, Lt);
//   move_point = p.move(FRONT, 2, Lx, Ly, Lz, Lt);
//   loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

//   for (int i = 0; i < Nc; i++) {
//     for (int j = 0; j < Nc; j++) {
//       // first row vector with col vector
//       temp = (src_local[0 * Nc + j] - src_local[2 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
//       dst_local[0 * Nc + i] += temp;
//       dst_local[2 * Nc + i] += temp * Complex(0, 1);
//       // second row vector with col vector
//       temp = (src_local[1 * Nc + j] + src_local[3 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
//       dst_local[1 * Nc + i] += temp;
//       dst_local[3 * Nc + i] += temp * Complex(0, -1);
//     }
//   }

//   move_point = p.move(BACK, 2, Lx, Ly, Lz, Lt);
//   loadGauge(u_local, gauge, 2, move_point, Lx, Ly, Lz, Lt);
//   loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

//   for (int i = 0; i < Nc; i++) {
//     for (int j = 0; j < Nc; j++) {
//       // first row vector with col vector
//       temp = (src_local[0 * Nc + j] + src_local[2 * Nc + j] * Complex(0, 1)) *
//              u_local[j * Nc + i].conj(); // transpose and conj
//       dst_local[0 * Nc + i] += temp;
//       dst_local[2 * Nc + i] += temp * Complex(0, -1);
//       // second row vector with col vector
//       temp = (src_local[1 * Nc + j] - src_local[3 * Nc + j] * Complex(0, 1)) *
//              u_local[j * Nc + i].conj(); // transpose and conj
//       dst_local[1 * Nc + i] += temp;
//       dst_local[3 * Nc + i] += temp * Complex(0, 1);
//     }
//   }

//   // \mu = 4
//   loadGauge(u_local, gauge, 3, p, Lx, Ly, Lz, Lt);
//   move_point = p.move(FRONT, 3, Lx, Ly, Lz, Lt);
//   loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

//   for (int i = 0; i < Nc; i++) {
//     for (int j = 0; j < Nc; j++) {
//       // first row vector with col vector
//       temp = (src_local[0 * Nc + j] - src_local[2 * Nc + j]) * u_local[i * Nc + j];
//       dst_local[0 * Nc + i] += temp;
//       dst_local[2 * Nc + i] += -temp;
//       // second row vector with col vector
//       temp = (src_local[1 * Nc + j] - src_local[3 * Nc + j]) * u_local[i * Nc + j];
//       dst_local[1 * Nc + i] += temp;
//       dst_local[3 * Nc + i] += -temp;
//     }
//   }

//   move_point = p.move(BACK, 3, Lx, Ly, Lz, Lt);
//   loadGauge(u_local, gauge, 3, move_point, Lx, Ly, Lz, Lt);
//   loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

//   for (int i = 0; i < Nc; i++) {
//     for (int j = 0; j < Nc; j++) {
//       // first row vector with col vector
//       temp = (src_local[0 * Nc + j] + src_local[2 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
//       dst_local[0 * Nc + i] += temp;
//       dst_local[2 * Nc + i] += temp;
//       // second row vector with col vector
//       temp = (src_local[1 * Nc + j] + src_local[3 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
//       dst_local[1 * Nc + i] += temp;
//       dst_local[3 * Nc + i] += temp;
//     }
//   }

//   // store result
//   double *dest = static_cast<double *>(fermion_out) + (blockIdx.x * BLOCK_SIZE) * Ns * Nc * 2;
//   double *dest_temp_double = (double *)dst_local;
//   for (int i = 0; i < Ns * Nc * 2; i++) {
//     shared_output_vec[threadIdx.x * Ns * Nc * 2 + i] = dest_temp_double[i];
//   }
//   __syncthreads();
//   // load to global memory
//   for (int i = threadIdx.x; i < BLOCK_SIZE * Ns * Nc * 2; i += BLOCK_SIZE) {
//     dest[i] = shared_output_vec[i];
//   }
// }

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


void MpiWilsonDslash::calculateDslash(int invert_flag) {
  int Lx = dslashParam_->Lx;
  int Ly = dslashParam_->Ly;
  int Lz = dslashParam_->Lz;
  int Lt = dslashParam_->Lt;
  int parity = dslashParam_->parity;

  int space = Lx * Ly * Lz * Lt >> 1;
  dim3 gridDim(space / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

  mpi_comm->preDslash(dslashParam_->fermion_in, parity, invert_flag);

  auto start = std::chrono::high_resolution_clock::now();
  void *args[] = {&dslashParam_->gauge, &dslashParam_->fermion_in, &dslashParam_->fermion_out, &Lx, &Ly, &Lz, &Lt, &parity, &grid_x, &grid_y, &grid_z, &grid_t};
  checkCudaErrors(cudaLaunchKernel((void *)mpiDslash, gridDim, blockDim, args));
  checkCudaErrors(cudaDeviceSynchronize());
  // boundary calculate
  mpi_comm->postDslash(dslashParam_->fermion_out, parity, invert_flag);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("total time: (without malloc free memcpy) : %.9lf sec\n", double(duration) / 1e9);
}


// void WilsonDslash();
// void callWilsonDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity) {
//   DslashParam dslash_param(fermion_in, fermion_out, gauge, param, parity);
//   WilsonDslash dslash_solver(dslash_param);
//   dslash_solver.calculateDslash();
// }
void callMpiWilsonDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag) {
  DslashParam dslash_param(fermion_in, fermion_out, gauge, param, parity);
  MpiWilsonDslash dslash_solver(dslash_param);
  dslash_solver.calculateDslash();
}


// this fermion in is the start addr of all
void wilsonDslashOneRound(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int invert_flag) {

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
    callMpiWilsonDslash(half_fermion_out, half_fermion_in, gauge, param, parity, invert_flag);
  }

  // checkCudaErrors(cudaFree(d_coeff));
  // checkCudaErrors(cudaFree(d_kappa));
}

void fullWilsonDslashOneRound (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int invert_flag) {
  // Dslash
  wilsonDslashOneRound(fermion_out, fermion_in, gauge, param, invert_flag);

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

