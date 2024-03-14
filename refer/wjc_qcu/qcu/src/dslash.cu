#include <cstdio>
#include <cmath>
#include <assert.h>
#include <chrono>
#include <mpi.h>
#include "qcu.h"
#include <cuda_runtime.h>
#include "qcu_complex.cuh"
#include "qcu_dslash.cuh"
#include "qcu_macro.cuh"
#include "qcu_complex_computation.cuh"
#include "qcu_point.cuh"
#include "qcu_communicator.cuh"
#include "qcu_clover_dslash.cuh"
#include "qcu_wilson_dslash_neo.cuh"
#include "qcu_wilson_dslash.cuh"
#include "qcu_shift_storage.cuh"
#include <iostream>
using std::cout;
using std::endl;
#define qcuPrint() { \
    printf("function %s line %d...\n", __FUNCTION__, __LINE__); \
}


void* qcu_gauge;
void loadQcuGauge(void* gauge, QcuParam *param) {
  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];

  checkCudaErrors(cudaMalloc(&qcu_gauge, sizeof(double) * Nd * Lx * Ly * Lz * Lt * (Nc-1) * Nc * 2));
  shiftGaugeStorage(qcu_gauge, gauge, TO_COALESCE, Lx, Ly, Lz, Lt);
}




void getDeviceInfo() {
  cudaDeviceProp deviceProp;
  int deviceCount;
  cudaError_t cudaError;
  cudaError = cudaGetDeviceCount(&deviceCount);
  for (int i = 0; i < deviceCount; i++) {
    cudaError = cudaGetDeviceProperties(&deviceProp, i);

    cout << "设备 " << i + 1 << " 的主要属性： " << endl;
    cout << "设备显卡型号： " << deviceProp.name << endl;
    cout << "设备全局内存总量（以MB为单位）： " << deviceProp.totalGlobalMem / 1024 / 1024 << endl; 
    cout << "设备上一个线程块（Block）中可用的最大共享内存（以KB为单位）： " << deviceProp.sharedMemPerBlock / 1024 << endl;  
    cout << "设备上一个线程块（Block）种可用的32位寄存器数量： " << deviceProp.regsPerBlock << endl;
    cout << "设备上一个线程块（Block）可包含的最大线程数量： " << deviceProp.maxThreadsPerBlock << endl;
    cout << "设备的计算功能集（Compute Capability）的版本号： " << deviceProp.major << "." << deviceProp.minor << endl;
    cout << "设备上多处理器的数量： " << deviceProp.multiProcessorCount << endl;
  }

}



void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity) {
  // getDeviceInfo();
  // parity ---- invert_flag

  // cloverDslashOneRound(fermion_out, fermion_in, gauge, param, 0);
  // cloverDslashOneRound(fermion_out, fermion_in, gauge, param, parity);
  // fullCloverDslashOneRound(fermion_out, fermion_in, gauge, param, 0);
  // wilsonDslashOneRound(fermion_out, fermion_in, gauge, param, parity);
  // callWilsonDslash(fermion_out, fermion_in, gauge, param, parity, 0);

  callWilsonDslash(fermion_out, fermion_in, gauge, param, parity, 0);

  callWilsonDslashNaive(fermion_out, fermion_in, gauge, param, parity, 0);
}
void fullDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int dagger_flag) {
  fullCloverDslashOneRound (fermion_out, fermion_in, gauge, param, dagger_flag);
}
