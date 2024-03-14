#pragma once



// 定义一个函数，用于求逆矩阵 6*6
//__device__ 
void inverseMatrix(Complex* matrix, Complex* result);

void gpu_saxpy(void* x, void* y, void* scalar, int vol);  // every point has Ns * Nc dim vector, y <- y + scalar * x, all addr are device address


void gpu_inner_product (void* x, void* y, void* result, int vol);

// xy inner product --->result (by partial result), vol means Lx * Ly * Lz * Lt
// void gpu_inner_product (void* x, void* y, void* result, void* partial_result, int vol); // partial_result: reduction space


#ifdef USE_MPI

#endif