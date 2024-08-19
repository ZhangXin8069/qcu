#include <cstdio>
#include <cstdlib>
#include <memory>
#include <chrono>
#include <cstdio>
void func(int, int) {}
class Timer {
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
  bool running;
  double elapsed_time_second;
public:
  Timer() = default;
  void start() {
    start_time = std::chrono::high_resolution_clock::now();
    running = true;
  }
  void stop() {
    end_time = std::chrono::high_resolution_clock::now();
    // elapsed_time = std::chrono::duration<double>(end_time -
    // start_time).count();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        end_time - start_time)
                        .count();
    elapsed_time_second = duration * 1e-9;
    running = false;
  }
  double getElapsedTimeSecond() {
    if (running) {
      stop();
    }
    return elapsed_time_second;
  }
};
inline double Tflops(size_t num, double time_second) {
  return num / (time_second * 1e12);
}
// inline void timerEvent (const char* msg, ) {
// }
// clang-format off
#define DEBUG
#ifdef DEBUG
#define TIMER_EVENT(cmd, repeat_time, msg)                                                    \
    do {                                                                                 \
        Timer timer;                                                                     \
        timer.start();                                                                   \
        cmd;                                                                             \
        timer.stop();                                                                    \
        printf("%s: %lf second\n", msg, timer.getElapsedTimeSecond() / repeat_time);                   \
    } while (0)
#else
#define TIMER_EVENT(cmd, repeat_time, msg) \
    do {                              \
        cmd;                          \
    } while (0)
#endif
// clang-format on
// printf("%s: Tflops = %lf\n", msg, timer.getElapsedTimeSecond() / (1e12));
#define _WARP_SIZE_ 32
#define PROFILE_DEBUG
#ifdef PROFILE_DEBUG
#define checkCudaErrors(ans)                                                        \
  do {                                                                         \
    cudaAssert((ans), __FILE__, __LINE__);                                     \
  } while (0)
inline void cudaAssert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
#else
#define checkCudaErrors(ans) ans
#endif
template <typename T> struct AddOp {
  __device__ __host__ __forceinline__ T operator()(const T &a,
                                                   const T &b) const {
    return a + b;
  }
};
template <typename T> struct SquareOp {
  __device__ __host__ __forceinline__ T operator()(const T &a) const {
    return a * a;
  }
};
template <template <typename> class ReductionOp, typename T>
__device__ __forceinline__ T warpReduce(T val) {
  for (int mask = _WARP_SIZE_ / 2; mask > 0; mask >>= 1) {
    val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}
// 把block reduce拆分为多个warp reduce来计算
// T为已经从global memory算完的，目前每个thread只对应一个结果
template <template <typename> class ReductionOp, typename T>
__device__ __forceinline__ void blockReduce(T val, T *smem) {
  int tid = threadIdx.x;
  int warp_id = tid / _WARP_SIZE_;
  int lane_id = tid & (_WARP_SIZE_ - 1);
  int warp_nums =
      (blockDim.x + _WARP_SIZE_ - 1) /
      _WARP_SIZE_; // 向上进1，以防分配的线程数量小于32导致warp nums为0
  val = warpReduce<ReductionOp, T>(val); // 先warp reduce
  if (lane_id == 0) {                    // TODO: 这个条件可能可以去掉
    smem[warp_id] = val;
  }
  __syncthreads();
  // 最后把每个warp的结果再作一个reduce得到最终一个block的结果
  T warp_val = tid < warp_nums ? smem[tid] : 0;
  T block_res = warpReduce<ReductionOp, T>(warp_val);
  __syncwarp();
  if (tid == 0) {
    smem[0] = block_res;
  }
}
// assumption: mrhs's m is not greater than 128
template <typename OutputType, typename InputType,
          template <typename> class ReductionOp>
__global__ void reduction_kernel(OutputType *output, const InputType *input,
                                 int vector_length) {
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;
  // 为了不损失精度，用输出类型进行规约操作
  ReductionOp<OutputType> reduce_op{};
  OutputType thread_sum{0};
  for (int i = global_id; i < vector_length; i += total_threads) {
    thread_sum = reduce_op(thread_sum, static_cast<OutputType>(input[i]));
  }
  __shared__ OutputType smem[64];
  blockReduce<ReductionOp>(thread_sum, smem);
  // __syncthreads();
  if (threadIdx.x == 0) {
    OutputType res = smem[0];
    output[blockIdx.x] = res;
  }
}
template <typename OutputType, typename InputType,
          template <typename> class ReduceOp>
void reduction_gpu_async(OutputType *output, OutputType *temp,
                         const InputType *input, int vector_length,
                         cudaStream_t stream) {
  int block_size = 256;
  int grid_size = (vector_length + block_size - 1) / block_size;
  reduction_kernel<OutputType, InputType, ReduceOp>
      <<<grid_size, block_size, 0, stream>>>(temp, input, vector_length);
  checkCudaErrors(cudaGetLastError());
  reduction_kernel<OutputType, OutputType, ReduceOp>
      <<<1, block_size, 0, stream>>>(output, temp, grid_size);
  checkCudaErrors(cudaGetLastError());
}
template <typename OutputType = double, typename InputType = double,
          template <typename> class ReduceOp = AddOp>
void profile_reduction_gpu_sync(OutputType *output, OutputType *temp,
                                const InputType *input, int vector_length,
                                cudaStream_t stream) {
  int block_size = 256;
  int grid_size = (vector_length + block_size - 1) / block_size;
  for (int i = 0; i < 1000; i++) {
    reduction_kernel<OutputType, InputType, ReduceOp>
        <<<grid_size, block_size, 0, stream>>>(temp, input, vector_length);
    checkCudaErrors(cudaGetLastError());
    reduction_kernel<OutputType, OutputType, ReduceOp>
        <<<1, block_size, 0, stream>>>(output, temp, grid_size);
    checkCudaErrors(cudaGetLastError());
  }
  checkCudaErrors(cudaDeviceSynchronize());
}
template <typename OutputType = double, typename InputType = double,
          template <typename> class ReduceOp = AddOp>
OutputType reduction_cpu(InputType *input, int vector_length) {
  OutputType res{0};
  ReduceOp<OutputType> reduce_op{};
  for (int i = 0; i < vector_length; i++) {
    res = reduce_op(res, static_cast<OutputType>(input[i]));
  }
  return res;
}
void init_host_data(double *data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = double(i % 10);
  }
}
int main() {
  using InputType = double;
  using OutputType = double;
  int size = 32 * 32 * 32 * 32;
  printf("size:%d", size);
  std::unique_ptr<InputType> h_input(new InputType[size]);
  std::unique_ptr<OutputType> h_output(new OutputType[size]);
  InputType *d_input;
  OutputType *d_output;
  OutputType *d_temp;
  OutputType d_res;
  checkCudaErrors(cudaMalloc(&d_input, size * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_output, sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_temp, size * sizeof(double)));
  init_host_data(h_input.get(), size);
  checkCudaErrors(cudaMemcpy(d_input, h_input.get(), size * sizeof(double),
                        cudaMemcpyHostToDevice));
  // cpu
  double res;
  TIMER_EVENT(res = reduction_cpu(h_input.get(), size), 1, "reduction_cpu");
  // gpu
  // warm up
  profile_reduction_gpu_sync<double, double, AddOp>(d_output, d_temp, d_input,
                                                    size, 0);
  const char *msg = "reduction_gpu";
  TIMER_EVENT(profile_reduction_gpu_sync(d_output, d_temp, d_input, size, 0),
              // func(1, 1),
              1000, msg);
  checkCudaErrors(
      cudaMemcpy(&d_res, d_output, sizeof(double), cudaMemcpyDeviceToHost));
  printf("gpu res: %lf, cpu res: %lf\n", d_res, res);
  // free
  checkCudaErrors(cudaFree(d_input));
  checkCudaErrors(cudaFree(d_output));
  checkCudaErrors(cudaFree(d_temp));
}