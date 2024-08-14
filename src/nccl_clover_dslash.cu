#include "../include/qcu.h"
#ifdef NCCL_CLOVER_DSLASH
void dslashCloverQcu(void *fermion_out, void *fermion_in, void *gauge,
                     QcuParam *param, int parity) {
  // define for nccl_clover_dslash
  LatticeSet _set;
  _set.give(param->lattice_size);
  _set.init();
  dptzyxcc2ccdptzyx(gauge, &_set);
  tzyxsc2sctzyx(fermion_in, &_set);
  tzyxsc2sctzyx(fermion_out, &_set);
  LatticeDslash _dslash;
  _dslash.give(&_set);
  void *clover;
  checkCudaErrors(cudaMalloc(&clover, (_set.lat_4dim * _LAT_SCSC_) *
                                          sizeof(LatticeComplex)));
  cudaError_t err;
  dim3 gridDim(_set.lat_4dim / _BLOCK_SIZE_);
  dim3 blockDim(_BLOCK_SIZE_);
  {
    // wilson dslash
    _dslash.run_test(fermion_out, fermion_in, gauge, parity);
  }
  {
    // make clover
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    make_clover<<<gridDim, blockDim>>>(gauge, clover, _set.device_lat_xyzt,
                                       parity);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    err = cudaGetLastError();
    checkCudaErrors(err);
    printf("make clover total time: (without malloc free memcpy) :%.9lf sec\n ",
           double(duration) / 1e9);
  }
  {
    // inverse clover
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    inverse_clover<<<gridDim, blockDim>>>(clover, _set.device_lat_xyzt);
    checkCudaErrors(cudaDeviceSynchronize());
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
  {
    // give clover
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    give_clover<<<gridDim, blockDim>>>(clover, fermion_out,
                                       _set.device_lat_xyzt);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    err = cudaGetLastError();
    checkCudaErrors(err);
    printf("give clover total time: (without malloc free memcpy) :%.9lf sec\n ",
           double(duration) / 1e9);
  }
  ccdptzyx2dptzyxcc(gauge, &_set);
  sctzyx2tzyxsc(fermion_in, &_set);
  sctzyx2tzyxsc(fermion_out, &_set);
  _set.end();
  // free
  checkCudaErrors(cudaFree(clover));
}
#endif