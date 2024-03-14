#include "qcu_complex_computation.cuh"
#include <cstdio>
#include "test_qcu_complex_computation.cuh"
#include "qcu_macro.cuh"
#include "qcu_complex.cuh"

static __global__ void initialize(void* a, void* b) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  double local_data[Ns * Nc * 2];
  for (int i = 0; i < Ns * Nc; i++) {
    local_data[2 * i + 0] = pos + 1;
    local_data[2 * i + 1] = pos + 2;
  }
  Complex* src = reinterpret_cast<Complex*>(local_data);
  Complex* dst_a = static_cast<Complex*>(a) + pos * Ns * Nc;
  Complex* dst_b = static_cast<Complex*>(b) + pos * Ns * Nc;
  for (int i = 0; i < Ns * Nc; i++) {
    dst_a[i] = src[i];
    dst_b[i] = src[i];
  }
}

static void saxpy_cpu(void* y, void* x, void* a, int vol) {
  Complex scalar = *(static_cast<Complex*>(a));
  Complex* x_ptr = static_cast<Complex*>(x);
  Complex* y_ptr = static_cast<Complex*>(y);

  for (int i = 0; i < vol * Nd * Nc; i++) {
    y_ptr[i] += scalar * x_ptr[i];
  }
}
// cpu function
static void check_saxpy(void* gpu_result, void* cpu_result, void* difference, int vol) {
  Complex* x_ptr = static_cast<Complex*>(gpu_result);
  Complex* y_ptr = static_cast<Complex*>(cpu_result);
  double* diff = static_cast<double*>(difference);
  double real_diff = 0;
  for (int i = 0; i < vol * Nd * Nc; i++) {
    real_diff += (x_ptr[i] - y_ptr[i]).norm2();
  }
  *diff = real_diff;
}

static void inner_product_cpu(void* h_a, void* h_b, void* h_res, int vol) {
  Complex* result = static_cast<Complex*>(h_res);
  result->clear2Zero();
  Complex* a_ptr = static_cast<Complex*>(h_a);
  Complex* b_ptr = static_cast<Complex*>(h_b);
  for (int i = 0; i < vol * Nd * Nc; i++) {
    *result += a_ptr[i] * b_ptr[i];
  }
}

static void test_inner_product() {
  constexpr int Lx = 16;
  constexpr int Ly = 16;
  constexpr int Lz = 16;
  constexpr int Lt = 32;
  constexpr int vol = Lx * Ly * Lz * Lt;
  int partial_length = vol / BLOCK_SIZE;
  void* h_a;
  void* h_b;
  void* d_a;
  void* d_b;
  void* partial_result;
  Complex h_cpu_result;
  Complex h_gpu_result;
  Complex* gpu_result;


  dim3 grid_dim(vol / BLOCK_SIZE);
  dim3 block_dim(BLOCK_SIZE);

  // malloc
  h_a = malloc(sizeof(Complex) * vol * Ns * Nc);
  h_b = malloc(sizeof(Complex) * vol * Ns * Nc);
  checkCudaErrors(cudaMalloc(&d_a, sizeof(Complex) * vol * Ns * Nc));
  checkCudaErrors(cudaMalloc(&d_b, sizeof(Complex) * vol * Ns * Nc));
  checkCudaErrors(cudaMalloc(&partial_result, sizeof(Complex) * partial_length));
  checkCudaErrors(cudaMalloc(&gpu_result, sizeof(Complex)));

  initialize<<<grid_dim, block_dim>>>(d_a, d_b);
  checkCudaErrors(cudaDeviceSynchronize());


  checkCudaErrors(cudaMemcpy(h_a, d_a, sizeof(Complex) * vol * Ns * Nc, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_b, d_b, sizeof(Complex) * vol * Ns * Nc, cudaMemcpyDeviceToHost));

  gpu_inner_product(d_a, d_b, static_cast<void*>(gpu_result), partial_result, vol);
  inner_product_cpu(h_a, h_b, static_cast<void*>(&h_cpu_result), vol);
  checkCudaErrors(cudaMemcpy(&h_gpu_result, gpu_result, sizeof(Complex), cudaMemcpyDeviceToHost));

  double difference = (h_cpu_result - h_gpu_result).norm2();

  printf("difference between cpu and gpu result %lf\n", difference);

  // free
  free(h_a);
  free(h_b);
  checkCudaErrors(cudaFree(d_a));
  checkCudaErrors(cudaFree(d_b));
  checkCudaErrors(cudaFree(partial_result));
  checkCudaErrors(cudaFree(gpu_result));
}


static void test_saxpy () {
  constexpr int Lx = 16;
  constexpr int Ly = 16;
  constexpr int Lz = 16;
  constexpr int Lt = 32;
  constexpr int vol = Lx * Ly * Lz * Lt;
  void* h_a;
  void* h_b;
  void* d_a;
  void* d_b;
  void* gpu_result;
  void* d_scalar;

  dim3 grid_dim(vol / BLOCK_SIZE);
  dim3 block_dim(BLOCK_SIZE);
  double scalar[2] = {1.0, 2.0};

  // malloc
  h_a = malloc(sizeof(Complex) * vol * Ns * Nc);
  h_b = malloc(sizeof(Complex) * vol * Ns * Nc);
  gpu_result = malloc(sizeof(Complex) * vol * Ns * Nc);
  checkCudaErrors(cudaMalloc(&d_a, sizeof(Complex) * vol * Ns * Nc));
  checkCudaErrors(cudaMalloc(&d_b, sizeof(Complex) * vol * Ns * Nc));
  checkCudaErrors(cudaMalloc(&d_scalar, sizeof(Complex)));

  initialize<<<grid_dim, block_dim>>>(d_a, d_b);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(h_a, d_a, sizeof(Complex) * vol * Ns * Nc, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_b, d_b, sizeof(Complex) * vol * Ns * Nc, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(d_scalar, scalar, sizeof(Complex), cudaMemcpyHostToDevice));
  // static void saxpy_cpu(void* y, void* x, void* a, int vol)
  gpu_saxpy(d_a, d_b, d_scalar, vol);
  saxpy_cpu(h_b, h_a, static_cast<void*>(&scalar), vol);
  // checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(gpu_result, d_b, sizeof(Complex) * vol * Ns * Nc, cudaMemcpyDeviceToHost));

  double difference = 0;
  // check_saxpy(void* gpu_result, void* cpu_result, void* difference)
  check_saxpy(h_b, gpu_result, static_cast<void*>(&difference), vol);
  // check_saxpy(h_b, h_a, static_cast<void*>(&difference), vol);

  printf("difference between cpu and gpu result %lf\n", difference);

  // free
  free(h_a);
  free(h_b);
  free(gpu_result);
  checkCudaErrors(cudaFree(d_a));
  checkCudaErrors(cudaFree(d_b));
  checkCudaErrors(cudaFree(d_scalar));
}


void test_computation() {
  printf("Now: testing saxpy...\n");
  test_saxpy();
  printf("Now: testing inner product...\n");
  test_inner_product();
}