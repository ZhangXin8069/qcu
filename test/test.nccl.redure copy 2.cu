#include <cstdio>
#include <cstdlib>
#include <memory>
#include <chrono>
#include <cstdio>
// 假设有一个 Complex 类定义如下
class Complex {
public:
    double real, imag;
    __device__ __host__ Complex() : real(0), imag(0) {}
    __device__ __host__ Complex(double r, double i) : real(r), imag(i) {}
    __device__ __host__ Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }
};
// Timer 类定义
class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
    bool running;
    double elapsed_time_second;
public:
    Timer() : running(false), elapsed_time_second(0) {}
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        running = true;
    }
    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
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
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#else
#define checkCudaErrors(ans) ans
#endif
// 用于 Complex 类型的加法操作符
struct AddOp {
    __device__ __host__ __forceinline__ Complex operator()(const Complex &a, const Complex &b) const {
        return a + b;
    }
};
__device__ __forceinline__ Complex warpReduce(Complex val) {
    for (int mask = _WARP_SIZE_ / 2; mask > 0; mask >>= 1) {
    Complex other;
    other.real = __shfl_xor_sync(0xffffffff, val.real, mask);
    other.imag = __shfl_xor_sync(0xffffffff, val.imag, mask);
    val = AddOp()(val, other);
    }
    return val;
}
__device__ __forceinline__ void blockReduce(Complex val, Complex *smem) {
    int tid = threadIdx.x;
    int warp_id = tid / _WARP_SIZE_;
    int lane_id = tid & (_WARP_SIZE_ - 1);
    int warp_nums = (blockDim.x + _WARP_SIZE_ - 1) / _WARP_SIZE_; 
    val = warpReduce(val);
    if (lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    Complex warp_val = tid < warp_nums ? smem[tid] : Complex();
    Complex block_res = warpReduce(warp_val);
    __syncwarp();
    if (tid == 0) {
        smem[0] = block_res;
    }
}
__global__ void reduction_kernel(Complex *output, const Complex *input, int vector_length) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    AddOp reduce_op{};
    Complex thread_sum(0.0, 0.0);
    for (int i = global_id; i < vector_length; i += total_threads) {
        thread_sum = reduce_op(thread_sum, input[i]);
    }
    __shared__ Complex smem[64];
    blockReduce(thread_sum, smem);
    if (threadIdx.x == 0) {
        output[blockIdx.x] = smem[0];
    }
}
void reduction_gpu_async(Complex *output, Complex *temp, const Complex *input, int vector_length, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (vector_length + block_size - 1) / block_size;
    reduction_kernel<<<grid_size, block_size, 0, stream>>>(temp, input, vector_length);
    checkCudaErrors(cudaGetLastError());
    reduction_kernel<<<1, block_size, 0, stream>>>(output, temp, grid_size);
    checkCudaErrors(cudaGetLastError());
}
void profile_reduction_gpu_sync(Complex *output, Complex *temp, const Complex *input, int vector_length, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (vector_length + block_size - 1) / block_size;
    for (int i = 0; i < 1000; i++) {
        reduction_kernel<<<grid_size, block_size, 0, stream>>>(temp, input, vector_length);
        checkCudaErrors(cudaGetLastError());
        reduction_kernel<<<1, block_size, 0, stream>>>(output, temp, grid_size);
        checkCudaErrors(cudaGetLastError());
    }
    checkCudaErrors(cudaDeviceSynchronize());
}
Complex reduction_cpu(const Complex *input, int vector_length) {
    Complex res(0.0, 0.0);
    AddOp reduce_op{};
    for (int i = 0; i < vector_length; i++) {
        res = reduce_op(res, input[i]);
    }
    return res;
}
void init_host_data(Complex *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = Complex(double(i % 10), double(i % 5));
    }
}
int main() {
    int size = 32 * 32 * 32 * 32;
    printf("size:%d", size);
    std::unique_ptr<Complex[]> h_input(new Complex[size]);
    std::unique_ptr<Complex[]> h_output(new Complex[size]);
    Complex *d_input;
    Complex *d_output;
    Complex *d_temp;
    Complex d_res;
    checkCudaErrors(cudaMalloc(&d_input, size * sizeof(Complex)));
    checkCudaErrors(cudaMalloc(&d_output, sizeof(Complex)));
    checkCudaErrors(cudaMalloc(&d_temp, size * sizeof(Complex)));
    init_host_data(h_input.get(), size);
    checkCudaErrors(cudaMemcpy(d_input, h_input.get(), size * sizeof(Complex), cudaMemcpyHostToDevice));
    // CPU reduction
    Complex res;
    TIMER_EVENT(res = reduction_cpu(h_input.get(), size), 1, "reduction_cpu");
    // GPU reduction
    // Warm up
    profile_reduction_gpu_sync(d_output, d_temp, d_input, size, 0);
    const char *msg = "reduction_gpu";
    TIMER_EVENT(profile_reduction_gpu_sync(d_output, d_temp, d_input, size, 0), 1000, msg);
    checkCudaErrors(cudaMemcpy(&d_res, d_output, sizeof(Complex), cudaMemcpyDeviceToHost));
    printf("GPU res: (%lf, %lf), CPU res: (%lf, %lf)\n", d_res.real, d_res.imag, res.real, res.imag);
    // Free
    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(d_temp));
    return 0;
}