#include "../include/qcu.h"
#ifdef LATTICE_CUDA
__global__ void give_random_value(void *device_random_value,
                                  unsigned long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *random_value =
      static_cast<LatticeComplex *>(device_random_value);
  curandState state_real, state_imag;
  curand_init(seed, idx, 0, &state_real);
  curand_init(seed, idx, 1, &state_imag);
  for (int i = 0; i < _LAT_SC_; ++i) {
    random_value[idx * _LAT_SC_ + i].real = curand_uniform(&state_real);
    random_value[idx * _LAT_SC_ + i].imag = curand_uniform(&state_imag);
  }
}
__global__ void give_custom_value(void *device_custom_value, double real,
                                  double imag) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *custom_value =
      static_cast<LatticeComplex *>(device_custom_value);
  for (int i = 0; i < _LAT_SC_; ++i) {
    custom_value[idx * _LAT_SC_ + i].real = real;
    custom_value[idx * _LAT_SC_ + i].imag = imag;
  }
}
__global__ void give_1zero(void *device_vals, const int vals_index) {
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex _(0.0, 0.0);
  origin_vals[vals_index] = _;
}
__global__ void give_1one(void *device_vals, const int vals_index) {
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex _(1.0, 0.0);
  origin_vals[vals_index] = _;
}
__global__ void give_1custom(void *device_vals, const int vals_index,
                             double real, double imag) {
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex _(real, imag);
  origin_vals[vals_index] = _;
}
__global__ void part_dot(void *device_vec0, void *device_vec1,
                         void *device_dot_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *origin_vec0 =
      (static_cast<LatticeComplex *>(device_vec0) + idx * _LAT_SC_);
  LatticeComplex *origin_vec1 =
      (static_cast<LatticeComplex *>(device_vec1) + idx * _LAT_SC_);
  LatticeComplex *origin_dot_vec =
      static_cast<LatticeComplex *>(device_dot_vec);
  LatticeComplex vec0[_LAT_SC_];
  LatticeComplex vec1[_LAT_SC_];
  LatticeComplex _(0.0, 0.0);
  give_ptr(vec0, origin_vec0, _LAT_SC_);
  give_ptr(vec1, origin_vec1, _LAT_SC_);
  for (int i = 0; i < _LAT_SC_; ++i) {
    _ += vec0[i].conj() * vec1[i];
  }
  origin_dot_vec[idx] = _;
}
__global__ void part_cut(void *device_vec0, void *device_vec1,
                         void *device_dot_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *origin_vec0 =
      (static_cast<LatticeComplex *>(device_vec0) + idx * _LAT_SC_);
  LatticeComplex *origin_vec1 =
      (static_cast<LatticeComplex *>(device_vec1) + idx * _LAT_SC_);
  LatticeComplex *origin_dot_vec =
      static_cast<LatticeComplex *>(device_dot_vec);
  LatticeComplex vec0[_LAT_SC_];
  LatticeComplex vec1[_LAT_SC_];
  LatticeComplex _(0.0, 0.0);
  give_ptr(vec0, origin_vec0, _LAT_SC_);
  give_ptr(vec1, origin_vec1, _LAT_SC_);
  for (int i = 0; i < _LAT_SC_; ++i) {
    _ += vec0[i] - vec1[i];
  }
  origin_dot_vec[idx] = _;
}
/*
Copy from WangJianCheng.
*/
struct AddOp {
  __device__ __host__ __forceinline__ LatticeComplex
  operator()(const LatticeComplex &a, const LatticeComplex &b) const {
    return a + b;
  }
};
__device__ __forceinline__ LatticeComplex warpReduce(LatticeComplex val) {
  for (int mask = _WARP_SIZE_ / 2; mask > 0; mask >>= 1) {
    LatticeComplex other;
    other.real = __shfl_xor_sync(0xffffffff, val.real, mask);
    other.imag = __shfl_xor_sync(0xffffffff, val.imag, mask);
    val = AddOp()(val, other);
  }
  return val;
}
__device__ __forceinline__ void blockReduce(LatticeComplex val,
                                            LatticeComplex *smem) {
  int tid = threadIdx.x;
  int warp_id = tid / _WARP_SIZE_;
  int lane_id = tid & (_WARP_SIZE_ - 1);
  int warp_nums = (blockDim.x + _WARP_SIZE_ - 1) / _WARP_SIZE_;
  val = warpReduce(val);
  if (lane_id == 0) {
    smem[warp_id] = val;
  }
  __syncthreads();
  LatticeComplex warp_val = tid < warp_nums ? smem[tid] : LatticeComplex();
  LatticeComplex block_res = warpReduce(warp_val);
  __syncwarp();
  if (tid == 0) {
    smem[0] = block_res;
  }
}
__global__ void reduction_kernel(LatticeComplex *output,
                                 const LatticeComplex *input,
                                 int vector_length) {
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;
  AddOp reduce_op{};
  LatticeComplex thread_sum(0.0, 0.0);
  for (int i = global_id; i < vector_length; i += total_threads) {
    thread_sum = reduce_op(thread_sum, input[i]);
  }
  __shared__ LatticeComplex smem[64];
  blockReduce(thread_sum, smem);
  if (threadIdx.x == 0) {
    output[blockIdx.x] = smem[0];
  }
}
void reduction_gpu_async(LatticeComplex *output, LatticeComplex *temp,
                         const LatticeComplex *input, int vector_length,
                         cudaStream_t stream) {
  int grid_size = (vector_length + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_;
  reduction_kernel<<<grid_size, _BLOCK_SIZE_, 0, stream>>>(temp, input,
                                                           vector_length);
  reduction_kernel<<<1, _BLOCK_SIZE_, 0, stream>>>(output, temp, grid_size);
}
void profile_reduction_gpu_sync(LatticeComplex *output, LatticeComplex *temp,
                                const LatticeComplex *input, int vector_length,
                                cudaStream_t stream) {
  int grid_size = (vector_length + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_;
  for (int i = 0; i < 1000; i++) {
    reduction_kernel<<<grid_size, _BLOCK_SIZE_, 0, stream>>>(temp, input,
                                                             vector_length);
    checkCudaErrors(cudaGetLastError());
    reduction_kernel<<<1, _BLOCK_SIZE_, 0, stream>>>(output, temp, grid_size);
    checkCudaErrors(cudaGetLastError());
  }
  checkCudaErrors(cudaDeviceSynchronize());
}

void perf_part_reduce(void *device_src_vec, void *device_dest_val,
                      void *device_tmp_vec, int size, cudaStream_t stream) {
  LatticeComplex *origin_src_vec =
      static_cast<LatticeComplex *>(device_src_vec);
  LatticeComplex *origin_dest_val =
      static_cast<LatticeComplex *>(device_dest_val);
  LatticeComplex *origin_tmp_vec =
      static_cast<LatticeComplex *>(device_tmp_vec);
  profile_reduction_gpu_sync(origin_dest_val, origin_tmp_vec, origin_src_vec,
                             size, stream);
}
void part_reduce(void *device_src_vec, void *device_dest_val,
                 void *device_tmp_vec, int size, cudaStream_t stream) {
  LatticeComplex *origin_src_vec =
      static_cast<LatticeComplex *>(device_src_vec);
  LatticeComplex *origin_dest_val =
      static_cast<LatticeComplex *>(device_dest_val);
  LatticeComplex *origin_tmp_vec =
      static_cast<LatticeComplex *>(device_tmp_vec);
  reduction_gpu_async(origin_dest_val, origin_tmp_vec, origin_src_vec, size,
                      stream);
}
/*
Copy from WangJianCheng.
*/

__global__ void _tzyxsc2sctzyx(void *device_fermi, void *device___fermi,
                               int lat_4dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *fermion =
      ((static_cast<LatticeComplex *>(device_fermi)) + idx * _LAT_SC_);
  LatticeComplex *_fermi =
      ((static_cast<LatticeComplex *>(device___fermi)) + idx);
  for (int i = 0; i < _LAT_SC_; i++) {
    _fermi[i * lat_4dim] = fermion[i];
  }
}
__global__ void _sctzyx2tzyxsc(void *device_fermi, void *device___fermi,
                               int lat_4dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *fermion =
      ((static_cast<LatticeComplex *>(device_fermi)) + idx);
  LatticeComplex *_fermi =
      ((static_cast<LatticeComplex *>(device___fermi)) + idx * _LAT_SC_);
  for (int i = 0; i < _LAT_SC_; i++) {
    _fermi[i] = fermion[i * lat_4dim];
  }
}
void tzyxsc2sctzyx(void *fermion, LatticeSet *set_ptr) {
  void *_fermi;
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  checkCudaErrors(cudaMallocAsync(
      &_fermi, set_ptr->lat_4dim_SC * sizeof(LatticeComplex), set_ptr->stream));
  _tzyxsc2sctzyx<<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
      fermion, _fermi, set_ptr->lat_4dim);
  CUBLAS_CHECK(
      cublasDcopy(set_ptr->cublasH,
                  set_ptr->lat_4dim_SC * sizeof(data_type) / sizeof(double),
                  (double *)_fermi, 1, (double *)fermion, 1));
  checkCudaErrors(cudaFreeAsync(_fermi, set_ptr->stream));
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
}
void sctzyx2tzyxsc(void *fermion, LatticeSet *set_ptr) {
  void *_fermi;
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  checkCudaErrors(cudaMallocAsync(
      &_fermi, set_ptr->lat_4dim_SC * sizeof(LatticeComplex), set_ptr->stream));
  _sctzyx2tzyxsc<<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
      fermion, _fermi, set_ptr->lat_4dim);
  CUBLAS_CHECK(
      cublasDcopy(set_ptr->cublasH,
                  set_ptr->lat_4dim_SC * sizeof(data_type) / sizeof(double),
                  (double *)_fermi, 1, (double *)fermion, 1));
  checkCudaErrors(cudaFreeAsync(_fermi, set_ptr->stream));
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
}
__global__ void _tzyxdcc2dcctzyx(void *device_gauge, void *device___gauge,
                                 int lat_4dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *gauge =
      ((static_cast<LatticeComplex *>(device_gauge)) + idx * _LAT_DCC_);
  LatticeComplex *_gauge =
      ((static_cast<LatticeComplex *>(device___gauge)) + idx);
  for (int i = 0; i < _LAT_DCC_; i++) {
    _gauge[i * lat_4dim] = gauge[i];
  }
}
__global__ void _dcctzyx2tzyxdcc(void *device_gauge, void *device___gauge,
                                 int lat_4dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *gauge = ((static_cast<LatticeComplex *>(device_gauge)) + idx);
  LatticeComplex *_gauge =
      ((static_cast<LatticeComplex *>(device___gauge)) + idx * _LAT_DCC_);
  for (int i = 0; i < _LAT_DCC_; i++) {
    _gauge[i] = gauge[i * lat_4dim];
  }
}
void tzyxdcc2dcctzyx(void *gauge, LatticeSet *set_ptr) {
  void *_gauge;
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  checkCudaErrors(
      cudaMallocAsync(&_gauge, set_ptr->lat_4dim_DCC * sizeof(LatticeComplex),
                      set_ptr->stream));
  _tzyxdcc2dcctzyx<<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
      gauge, _gauge, set_ptr->lat_4dim);
  CUBLAS_CHECK(
      cublasDcopy(set_ptr->cublasH,
                  set_ptr->lat_4dim_DCC * sizeof(data_type) / sizeof(double),
                  (double *)_gauge, 1, (double *)gauge, 1));
  checkCudaErrors(cudaFreeAsync(_gauge, set_ptr->stream));
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
}
void dcctzyx2tzyxdcc(void *gauge, LatticeSet *set_ptr) {
  void *_gauge;
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  checkCudaErrors(
      cudaMallocAsync(&_gauge, set_ptr->lat_4dim_DCC * sizeof(LatticeComplex),
                      set_ptr->stream));
  _dcctzyx2tzyxdcc<<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
      gauge, _gauge, set_ptr->lat_4dim);
  CUBLAS_CHECK(
      cublasDcopy(set_ptr->cublasH,
                  set_ptr->lat_4dim_DCC * sizeof(data_type) / sizeof(double),
                  (double *)_gauge, 1, (double *)gauge, 1));
  checkCudaErrors(cudaFreeAsync(_gauge, set_ptr->stream));
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
}
#endif