#include "../include/qcu.h"
#ifdef LATTICE_CUDA
__global__ void give_random_vals(void *device_random_vals,
                                  unsigned long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *random_vals =
      static_cast<LatticeComplex *>(device_random_vals);
  curandState state_real, state_imag;
  curand_init(seed, idx, 0, &state_real);
  curand_init(seed, idx, 1, &state_imag);
  for (int i = 0; i < _LAT_SC_; ++i) {
    random_vals[idx * _LAT_SC_ + i].real = curand_uniform(&state_real);
    random_vals[idx * _LAT_SC_ + i].imag = curand_uniform(&state_imag);
  }
}
__global__ void give_custom_vals(void *device_custom_vals, double real,
                                  double imag) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *custom_vals =
      static_cast<LatticeComplex *>(device_custom_vals);
  for (int i = 0; i < _LAT_SC_; ++i) {
    custom_vals[idx * _LAT_SC_ + i].real = real;
    custom_vals[idx * _LAT_SC_ + i].imag = imag;
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
__global__ void _tzyxsc2sctzyx(void *device_fermi, void *device__fermi,
                               int lat_4dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *fermion =
      ((static_cast<LatticeComplex *>(device_fermi)) + idx * _LAT_SC_);
  LatticeComplex *_fermion =
      ((static_cast<LatticeComplex *>(device__fermi)) + idx);
  for (int i = 0; i < _LAT_SC_; i++) {
    _fermion[i * lat_4dim] = fermion[i];
  }
}
__global__ void _sctzyx2tzyxsc(void *device_fermi, void *device__fermi,
                               int lat_4dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *fermion =
      ((static_cast<LatticeComplex *>(device_fermi)) + idx);
  LatticeComplex *_fermion =
      ((static_cast<LatticeComplex *>(device__fermi)) + idx * _LAT_SC_);
  for (int i = 0; i < _LAT_SC_; i++) {
    _fermion[i] = fermion[i * lat_4dim];
  }
}
void tzyxsc2sctzyx(void *fermion, LatticeSet *set_ptr) {
  checkCudaErrors(cudaDeviceSynchronize());
  void *_fermion;
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  checkCudaErrors(cudaMallocAsync(&_fermion,
                                  set_ptr->lat_4dim_SC * sizeof(LatticeComplex),
                                  set_ptr->stream));
  _tzyxsc2sctzyx<<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
      fermion, _fermion, set_ptr->lat_4dim);
  CUBLAS_CHECK(
      cublasDcopy(set_ptr->cublasH,
                  set_ptr->lat_4dim_SC * sizeof(data_type) / sizeof(double),
                  (double *)_fermion, 1, (double *)fermion, 1));
  checkCudaErrors(cudaFreeAsync(_fermion, set_ptr->stream));
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  checkCudaErrors(cudaDeviceSynchronize());
}
void sctzyx2tzyxsc(void *fermion, LatticeSet *set_ptr) {
  checkCudaErrors(cudaDeviceSynchronize());
  void *_fermion;
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  checkCudaErrors(cudaMallocAsync(&_fermion,
                                  set_ptr->lat_4dim_SC * sizeof(LatticeComplex),
                                  set_ptr->stream));
  _sctzyx2tzyxsc<<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
      fermion, _fermion, set_ptr->lat_4dim);
  CUBLAS_CHECK(
      cublasDcopy(set_ptr->cublasH,
                  set_ptr->lat_4dim_SC * sizeof(data_type) / sizeof(double),
                  (double *)_fermion, 1, (double *)fermion, 1));
  checkCudaErrors(cudaFreeAsync(_fermion, set_ptr->stream));
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  checkCudaErrors(cudaDeviceSynchronize());
}
__global__ void _dptzyxcc2ccdptzyx(void *device_gauge, void *device__gauge,
                                   int lat_4dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *gauge =
      ((static_cast<LatticeComplex *>(device_gauge)) + idx * _LAT_CC_);
  LatticeComplex *_gauge =
      ((static_cast<LatticeComplex *>(device__gauge)) + idx);
  for (int p = 0; p < _EVEN_ODD_; p++) {
    for (int d = 0; d < _LAT_D_; d++) {
      for (int cc = 0; cc < _LAT_CC_; cc++) {
        _gauge[((cc * _LAT_D_ + d) * _EVEN_ODD_ + p) * lat_4dim] =
            gauge[(d * _EVEN_ODD_ + p) * _LAT_CC_ * lat_4dim + cc];
      }
    }
  }
}
__global__ void _ccdptzyx2dptzyxcc(void *device_gauge, void *device__gauge,
                                   int lat_4dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *gauge = ((static_cast<LatticeComplex *>(device_gauge)) + idx);
  LatticeComplex *_gauge =
      ((static_cast<LatticeComplex *>(device__gauge)) + idx * _LAT_CC_);
  for (int p = 0; p < _EVEN_ODD_; p++) {
    for (int d = 0; d < _LAT_D_; d++) {
      for (int cc = 0; cc < _LAT_CC_; cc++) {
        _gauge[(d * _EVEN_ODD_ + p) * _LAT_CC_ * lat_4dim + cc] =
            gauge[((cc * _LAT_D_ + d) * _EVEN_ODD_ + p) * lat_4dim];
      }
    }
  }
}
void dptzyxcc2ccdptzyx(void *gauge, LatticeSet *set_ptr) {
  checkCudaErrors(cudaDeviceSynchronize());
  void *_gauge;
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  checkCudaErrors(cudaMallocAsync(
      &_gauge, set_ptr->lat_4dim_DCC * _EVEN_ODD_ * sizeof(LatticeComplex),
      set_ptr->stream));
  _dptzyxcc2ccdptzyx<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                       set_ptr->stream>>>(gauge, _gauge, set_ptr->lat_4dim);
  CUBLAS_CHECK(cublasDcopy(set_ptr->cublasH,
                           set_ptr->lat_4dim_DCC * _EVEN_ODD_ *
                               sizeof(data_type) / sizeof(double),
                           (double *)_gauge, 1, (double *)gauge, 1));
  checkCudaErrors(cudaFreeAsync(_gauge, set_ptr->stream));
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  checkCudaErrors(cudaDeviceSynchronize());
}
void ccdptzyx2dptzyxcc(void *gauge, LatticeSet *set_ptr) {
  checkCudaErrors(cudaDeviceSynchronize());
  void *_gauge;
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  checkCudaErrors(cudaMallocAsync(
      &_gauge, set_ptr->lat_4dim_DCC * _EVEN_ODD_ * sizeof(LatticeComplex),
      set_ptr->stream));
  _ccdptzyx2dptzyxcc<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                       set_ptr->stream>>>(gauge, _gauge, set_ptr->lat_4dim);
  CUBLAS_CHECK(cublasDcopy(set_ptr->cublasH,
                           set_ptr->lat_4dim_DCC * _EVEN_ODD_ *
                               sizeof(data_type) / sizeof(double),
                           (double *)_gauge, 1, (double *)gauge, 1));
  checkCudaErrors(cudaFreeAsync(_gauge, set_ptr->stream));
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void _ptzyxsc2psctzyx(void *device_fermi, void *device__fermi,
                                 int lat_4dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *fermion =
      ((static_cast<LatticeComplex *>(device_fermi)) + idx * _LAT_SC_);
  LatticeComplex *_fermion =
      ((static_cast<LatticeComplex *>(device__fermi)) + idx);
  for (int p = 0; p < _EVEN_ODD_; p++) {
    for (int i = 0; i < _LAT_SC_; i++) {
      _fermion[(p * _LAT_SC_ + i) * lat_4dim] =
          fermion[p * _LAT_SC_ * lat_4dim + i];
    }
  }
}
__global__ void _psctzyx2ptzyxsc(void *device_fermi, void *device__fermi,
                                 int lat_4dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *fermion =
      ((static_cast<LatticeComplex *>(device_fermi)) + idx);
  LatticeComplex *_fermion =
      ((static_cast<LatticeComplex *>(device__fermi)) + idx * _LAT_SC_);
  for (int p = 0; p < _EVEN_ODD_; p++) {
    for (int i = 0; i < _LAT_SC_; i++) {
      _fermion[p * _LAT_SC_ * lat_4dim + i] =
          fermion[(p * _LAT_SC_ + i) * lat_4dim];
    }
  }
}
void ptzyxsc2psctzyx(void *fermion, LatticeSet *set_ptr) {
  checkCudaErrors(cudaDeviceSynchronize());
  void *_fermion;
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  checkCudaErrors(cudaMallocAsync(
      &_fermion, set_ptr->lat_4dim_SC * _EVEN_ODD_ * sizeof(LatticeComplex),
      set_ptr->stream));
  _ptzyxsc2psctzyx<<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
      fermion, _fermion, set_ptr->lat_4dim);
  CUBLAS_CHECK(cublasDcopy(set_ptr->cublasH,
                           set_ptr->lat_4dim_SC * _EVEN_ODD_ *
                               sizeof(data_type) / sizeof(double),
                           (double *)_fermion, 1, (double *)fermion, 1));
  checkCudaErrors(cudaFreeAsync(_fermion, set_ptr->stream));
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  checkCudaErrors(cudaDeviceSynchronize());
}
void psctzyx2ptzyxsc(void *fermion, LatticeSet *set_ptr) {
  checkCudaErrors(cudaDeviceSynchronize());
  void *_fermion;
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  checkCudaErrors(cudaMallocAsync(
      &_fermion, set_ptr->lat_4dim_SC * _EVEN_ODD_ * sizeof(LatticeComplex),
      set_ptr->stream));
  _psctzyx2ptzyxsc<<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
      fermion, _fermion, set_ptr->lat_4dim);
  CUBLAS_CHECK(cublasDcopy(set_ptr->cublasH,
                           set_ptr->lat_4dim_SC * _EVEN_ODD_ *
                               sizeof(data_type) / sizeof(double),
                           (double *)_fermion, 1, (double *)fermion, 1));
  checkCudaErrors(cudaFreeAsync(_fermion, set_ptr->stream));
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  checkCudaErrors(cudaDeviceSynchronize());
}

#endif