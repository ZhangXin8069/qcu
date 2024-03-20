#include <cuda_runtime.h>
#include <iostream>

// CUDA 核函数，将矩阵乘以标量
__global__ void matrixScalarMultiply(float *matrix, float scalar, float *result,
                                     int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rows * cols) {
    for (int i = 0; i < 1000; ++i) { // 增加循环次数以增加计算量
      result[idx] = matrix[idx] * scalar;
    }
  }
}

int main() {
  const int rows = 1000;
  const int cols = 1000;
  const int numStreams = 4; // 定义流的数量

  float scalar = 2.0f;
  cudaEvent_t start, stop;

  // 分配统一内存给矩阵和结果
  float *matrix, *result;

  cudaMallocManaged(&matrix, rows * cols * sizeof(float));
  cudaMallocManaged(&result, rows * cols * sizeof(float));

  // 初始化输入数据
  for (int i = 0; i < rows * cols; ++i) {
    matrix[i] = i;
  }

  cudaStream_t stream[numStreams];

  // 创建流
  for (int i = 0; i < numStreams; ++i) {
    cudaStreamCreate(&stream[i]);
  }

  // 启动计时器
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // 启动核函数，使用多个CUDA流
  for (int i = 0; i < numStreams; ++i) {
    matrixScalarMultiply<<<(rows * cols) / 256 + 1, 256, 0, stream[i]>>>(
        matrix + i * (rows * cols / numStreams), scalar,
        result + i * (rows * cols / numStreams), rows, cols);
  }

  // 同步流，等待所有操作完成
  for (int i = 0; i < numStreams; ++i) {
    cudaStreamSynchronize(stream[i]);
    cudaStreamDestroy(stream[i]);
  }

  // 停止计时器并计算执行时间
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Execution time: " << milliseconds << " ms" << std::endl;

  // 打印部分结果
  for (int i = 0; i < 10; ++i) {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;

  // 释放内存
  cudaFree(matrix);
  cudaFree(result);

  return 0;
}
