#ifndef _LATTICE_CUDA_H
#define _LATTICE_CUDA_H
#include "./include.h"
#include "./lattice_set.h"

__global__ void give_random_value(void *device_random_value,
                                  unsigned long seed);

__global__ void give_custom_value(void *device_custom_value, double real,
                                  double imag);

__global__ void give_1zero(void *device_vals, const int vals_index);

__global__ void give_1one(void *device_vals, const int vals_index);

__global__ void give_1custom(void *device_vals, const int vals_index,
                             double real, double imag);

__global__ void part_dot(void *device_vec0, void *device_vec1,
                         void *device_dot_vec);

__global__ void part_cut(void *device_vec0, void *device_vec1,
                         void *device_dot_vec);

void perf_part_reduce(void *device_src_vec, void *device_dest_val,
                      void *device_tmp_vec, int size, cudaStream_t stream);

void part_reduce(void *device_src_vec, void *device_dest_val,
                 void *device_tmp_vec, int size, cudaStream_t stream);
// memory alignment
#define ALIGN_TO(A, B) (((A + B - 1) / B) * B)
// device memory pitch alignment
static const size_t device_alignment = 32;
// type traits
template <typename T> struct traits;
template <> struct traits<float> {
  // scalar type
  typedef float T;
  typedef T S;
  static constexpr T zero = 0.f;
  static constexpr cudaDataType cuda_data_type = CUDA_R_32F;
  inline static S abs(T val) { return fabs(val); }
  template <typename RNG> inline static T rand(RNG &gen) { return (S)gen(); }
  inline static T add(T a, T b) { return a + b; }
  inline static T mul(T v, double f) { return v * f; }
};
template <> struct traits<double> {
  // scalar type
  typedef double T;
  typedef T S;
  static constexpr T zero = 0.;
  static constexpr cudaDataType cuda_data_type = CUDA_R_64F;
  inline static S abs(T val) { return fabs(val); }
  template <typename RNG> inline static T rand(RNG &gen) { return (S)gen(); }
  inline static T add(T a, T b) { return a + b; }
  inline static T mul(T v, double f) { return v * f; }
};
template <> struct traits<cuFloatComplex> {
  // scalar type
  typedef float S;
  typedef cuFloatComplex T;
  static constexpr T zero = {0.f, 0.f};
  static constexpr cudaDataType cuda_data_type = CUDA_C_32F;
  inline static S abs(T val) { return cuCabsf(val); }
  template <typename RNG> inline static T rand(RNG &gen) {
    return make_cuFloatComplex((S)gen(), (S)gen());
  }
  inline static T add(T a, T b) { return cuCaddf(a, b); }
  inline static T add(T a, S b) {
    return cuCaddf(a, make_cuFloatComplex(b, 0.f));
  }
  inline static T mul(T v, double f) {
    return make_cuFloatComplex(v.x * f, v.y * f);
  }
};
template <> struct traits<cuDoubleComplex> {
  // scalar type
  typedef double S;
  typedef cuDoubleComplex T;
  static constexpr T zero = {0., 0.};
  static constexpr cudaDataType cuda_data_type = CUDA_C_64F;
  inline static S abs(T val) { return cuCabs(val); }
  template <typename RNG> inline static T rand(RNG &gen) {
    return make_cuDoubleComplex((S)gen(), (S)gen());
  }
  inline static T add(T a, T b) { return cuCadd(a, b); }
  inline static T add(T a, S b) {
    return cuCadd(a, make_cuDoubleComplex(b, 0.));
  }
  inline static T mul(T v, double f) {
    return make_cuDoubleComplex(v.x * f, v.y * f);
  }
};
/*
template <typename T>
void print_matrix(const int &m, const int &n, const T *A, const int &lda);
template <>
void print_matrix(const int &m, const int &n, const float *A, const int &lda) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::printf("%0.2f ", A[j * lda + i]);
    }
    std::printf("\n");
  }
}
template <>
void print_matrix(const int &m, const int &n, const double *A, const int &lda) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::printf("%0.2f ", A[j * lda + i]);
    }
    std::printf("\n");
  }
}
template <>
void print_matrix(const int &m, const int &n, const cuComplex *A,
                  const int &lda) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
    }
    std::printf("\n");
  }
}
template <>
void print_matrix(const int &m, const int &n, const cuDoubleComplex *A,
                  const int &lda) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
    }
    std::printf("\n");
  }
}
template <typename T>
void print_packed_matrix(cublasFillMode_t uplo, const int &n, const T *A);
template <>
void print_packed_matrix(cublasFillMode_t uplo, const int &n, const float *A) {
  size_t off = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if ((uplo == CUBLAS_FILL_MODE_UPPER && j >= i) ||
          (uplo == CUBLAS_FILL_MODE_LOWER && j <= i)) {
        std::printf("%6.2f ", A[off++]);
      } else if (uplo == CUBLAS_FILL_MODE_UPPER) {
        std::printf("       ");
      }
    }
    std::printf("\n");
  }
}
template <>
void print_packed_matrix(cublasFillMode_t uplo, const int &n, const double *A) {
  size_t off = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if ((uplo == CUBLAS_FILL_MODE_UPPER && j >= i) ||
          (uplo == CUBLAS_FILL_MODE_LOWER && j <= i)) {
        std::printf("%6.2f ", A[off++]);
      } else if (uplo == CUBLAS_FILL_MODE_UPPER) {
        std::printf("       ");
      }
    }
    std::printf("\n");
  }
}
template <>
void print_packed_matrix(cublasFillMode_t uplo, const int &n,
                         const cuComplex *A) {
  size_t off = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if ((uplo == CUBLAS_FILL_MODE_UPPER && j >= i) ||
          (uplo == CUBLAS_FILL_MODE_LOWER && j <= i)) {
        std::printf("%6.2f + %6.2fj ", A[off].x, A[off].y);
        off++;
      } else if (uplo == CUBLAS_FILL_MODE_UPPER) {
        std::printf("                 ");
      }
    }
    std::printf("\n");
  }
}
template <>
void print_packed_matrix(cublasFillMode_t uplo, const int &n,
                         const cuDoubleComplex *A) {
  size_t off = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if ((uplo == CUBLAS_FILL_MODE_UPPER && j >= i) ||
          (uplo == CUBLAS_FILL_MODE_LOWER && j <= i)) {
        std::printf("%6.2f + %6.2fj ", A[off].x, A[off].y);
        off++;
      } else if (uplo == CUBLAS_FILL_MODE_UPPER) {
        std::printf("                 ");
      }
    }
    std::printf("\n");
  }
}
template <typename T> void print_vector(const int &m, const T *A);
template <> void print_vector(const int &m, const float *A) {
  for (int i = 0; i < m; i++) {
    std::printf("%0.2f ", A[i]);
  }
  std::printf("\n");
}
template <> void print_vector(const int &m, const double *A) {
  for (int i = 0; i < m; i++) {
    std::printf("%0.2f ", A[i]);
  }
  std::printf("\n");
}
template <> void print_vector(const int &m, const cuComplex *A) {
  for (int i = 0; i < m; i++) {
    std::printf("%0.2f + %0.2fj ", A[i].x, A[i].y);
  }
  std::printf("\n");
}
template <> void print_vector(const int &m, const cuDoubleComplex *A) {
  for (int i = 0; i < m; i++) {
    std::printf("%0.2f + %0.2fj ", A[i].x, A[i].y);
  }
  std::printf("\n");
}
template <typename T>
void generate_random_matrix(int m, int n, T **A, int *lda) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<typename traits<T>::S> dis(-1.0, 1.0);
  auto rand_gen = std::bind(dis, gen);
  *lda = n;
  size_t matrix_mem_size = static_cast<size_t>(*lda * m * sizeof(T));
  // suppress gcc 7 size warning
  if (matrix_mem_size <= PTRDIFF_MAX)
    *A = (T *)malloc(matrix_mem_size);
  else
    throw std::runtime_error("Memory allocation size is too large");
  if (*A == NULL)
    throw std::runtime_error("Unable to allocate host matrix");
  // random matrix and accumulate row sums
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      T *A_row = (*A) + *lda * i;
      A_row[j] = traits<T>::rand(rand_gen);
    }
  }
}
// Makes matrix A of size mxn and leading dimension lda diagonal dominant
template <typename T>
void make_diag_dominant_matrix(int m, int n, T *A, int lda) {
  for (int i = 0; i < std::min(m, n); ++i) {
    T *A_row = A + lda * i;
    auto row_sum = traits<typename traits<T>::S>::zero;
    for (int j = 0; j < n; ++j) {
      row_sum += traits<T>::abs(A_row[j]);
    }
    A_row[i] = traits<T>::add(A_row[i], row_sum);
  }
}
// Returns cudaDataType value as defined in library_types.h for the string
// containing type name
cudaDataType get_cuda_library_type(std::string type_string) {
  if (type_string.compare("CUDA_R_16F") == 0)
    return CUDA_R_16F;
  else if (type_string.compare("CUDA_C_16F") == 0)
    return CUDA_C_16F;
  else if (type_string.compare("CUDA_R_32F") == 0)
    return CUDA_R_32F;
  else if (type_string.compare("CUDA_C_32F") == 0)
    return CUDA_C_32F;
  else if (type_string.compare("CUDA_R_64F") == 0)
    return CUDA_R_64F;
  else if (type_string.compare("CUDA_C_64F") == 0)
    return CUDA_C_64F;
  else if (type_string.compare("CUDA_R_8I") == 0)
    return CUDA_R_8I;
  else if (type_string.compare("CUDA_C_8I") == 0)
    return CUDA_C_8I;
  else if (type_string.compare("CUDA_R_8U") == 0)
    return CUDA_R_8U;
  else if (type_string.compare("CUDA_C_8U") == 0)
    return CUDA_C_8U;
  else if (type_string.compare("CUDA_R_32I") == 0)
    return CUDA_R_32I;
  else if (type_string.compare("CUDA_C_32I") == 0)
    return CUDA_C_32I;
  else if (type_string.compare("CUDA_R_32U") == 0)
    return CUDA_R_32U;
  else if (type_string.compare("CUDA_C_32U") == 0)
    return CUDA_C_32U;
  else
    throw std::runtime_error("Unknown CUDA datatype");
}
struct GPUTimer {
  GPUTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_, 0);
  }
  ~GPUTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  void start() { cudaEventRecord(start_, 0); }
  float seconds() {
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);
    float time;
    cudaEventElapsedTime(&time, start_, stop_);
    return time * 1e-3;
  }

private:
  cudaEvent_t start_, stop_;
};
/*
        // Set up timing
        GPUTimer timer;
        timer.start();
        HANDLE_ERROR(cutensorContract(handle,
                                      planJit,
                                      (void*) &alpha, A_d, B_d,
                                      (void*) &beta,  C_d, C_d,
                                      workJit, actualWorkspaceSizeJit, stream))
        // Synchronize and measure timing
        auto time = timer.seconds();
*/

__global__ void _tzyxsc2sctzyx(void *device_fermi, void *device___fermi,
                               int lat_4dim);
__global__ void _sctzyx2tzyxsc(void *device_fermi, void *device___fermi,
                               int lat_4dim);
void tzyxsc2sctzyx(void *fermion, LatticeSet *set_ptr);
void sctzyx2tzyxsc(void *fermion, LatticeSet *set_ptr);
__global__ void _dptzyxcc2ccdptzyx(void *device_gauge, void *device___gauge,
                                   int lat_4dim);
__global__ void _ccdptzyx2dptzyxcc(void *device_gauge, void *device___gauge,
                                   int lat_4dim);
void dptzyxcc2ccdptzyx(void *gauge, LatticeSet *set_ptr);
void ccdptzyx2dptzyxcc(void *gauge, LatticeSet *set_ptr);
__global__ void _ptzyxsc2psctzyx(void *device_fermi, void *device___fermi,
                                 int lat_4dim);
__global__ void _psctzyx2ptzyxsc(void *device_fermi, void *device___fermi,
                                 int lat_4dim);
void ptzyxsc2psctzyx(void *fermion, LatticeSet *set_ptr);
void psctzyx2ptzyxsc(void *fermion, LatticeSet *set_ptr);
#endif