#ifndef _LATTICE_CLOVER_DSLASH_H
#define _LATTICE_CLOVER_DSLASH_H
#include "./clover_dslash.h"
#include "./define.h"
#include "./lattice_set.h"
struct LatticeCloverDslash {
  LatticeSet *set_ptr;
  cudaError_t err;
  LatticeWilsonDslash wilson_dslash;
  void *clover;
  void give(LatticeSet *_set_ptr) { set_ptr = _set_ptr; }
  void init() {
    checkCudaErrors(cudaMallocAsync(
        &clover, (set_ptr->lat_4dim * _LAT_SCSC_) * sizeof(LatticeComplex),
        set_ptr->stream));
  }
  void make(void *gauge, int parity) {
    // make clover
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    auto start = std::chrono::high_resolution_clock::now();
    make_clover<<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        gauge, clover, set_ptr->device_lat_xyzt, parity);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    err = cudaGetLastError();
    checkCudaErrors(err);
    printf("make clover total time: (without malloc free memcpy) :%.9lf sec\n ",
           double(duration) / 1e9);
  }
  void inverse() {
    // inverse clover
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    auto start = std::chrono::high_resolution_clock::now();
    inverse_clover<<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        clover, set_ptr->device_lat_xyzt);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    err = cudaGetLastError();
    checkCudaErrors(err);
    printf(
        "inverse clover total time: (without malloc free memcpy) :%.9lf sec\n ",
        double(duration) / 1e9);
  }
  void give(void *fermion_out) {
    // give clover
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    auto start = std::chrono::high_resolution_clock::now();
    give_clover<<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        clover, fermion_out, set_ptr->device_lat_xyzt);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    err = cudaGetLastError();
    checkCudaErrors(err);
    printf("give clover total time: (without malloc free memcpy) :%.9lf sec\n ",
           double(duration) / 1e9);
  }
  void end() {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(clover, set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
};
#endif
