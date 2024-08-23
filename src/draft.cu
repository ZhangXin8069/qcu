#include "../include/qcu.h"
#ifdef DRAFT
void dslashQcu(void *fermion_out, void *fermion_in, void *gauge,
               QcuParam *param, int parity) {
  // define for wilson_dslash
  LatticeSet _set;
  _set.give(param->lattice_size);
  _set.init();
  dptzyxcc2ccdptzyx(gauge, &_set);
  tzyxsc2sctzyx(fermion_in, &_set);
  tzyxsc2sctzyx(fermion_out, &_set);
  auto start = std::chrono::high_resolution_clock::now();
  wilson_dslash<<<_set.gridDim, _set.blockDim>>>(gauge, fermion_in, fermion_out,
                                                 _set.device_lat_xyzt, parity);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  cudaError_t err = cudaGetLastError();
  checkCudaErrors(err);
  printf("wilson dslash total time: (without malloc free memcpy) :%.9lf "
         "sec\n",
         double(duration) / 1e9);
  ccdptzyx2dptzyxcc(gauge, &_set);
  sctzyx2tzyxsc(fermion_in, &_set);
  sctzyx2tzyxsc(fermion_out, &_set);
  _set.end();
}
void mpiDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                  QcuParam *param, int parity, QcuParam *grid) {}
void mpiBistabCgQcu(void *gauge, QcuParam *param, QcuParam *grid) {}
void dslashCloverQcu(void *fermion_out, void *fermion_in, void *gauge,
                     QcuParam *param, int parity) {
  // define for nccl_clover_dslash
  LatticeSet _set;
  _set.give(param->lattice_size);
  _set.init();
  dptzyxcc2ccdptzyx(gauge, &_set);
  tzyxsc2sctzyx(fermion_in, &_set);
  tzyxsc2sctzyx(fermion_out, &_set);
  LatticeWilsonDslash _wilson_dslash;
  _wilson_dslash.give(&_set);
  void *clover;
  checkCudaErrors(cudaMallocAsync(
      &clover, (_set.lat_4dim * _LAT_SCSC_) * sizeof(LatticeComplex),
      _set.stream));
  cudaError_t err;
  {
    // wilson dslash
    _wilson_dslash.run_test(fermion_out, fermion_in, gauge, parity);
  }
  {
    // make clover
    checkCudaErrors(cudaStreamSynchronize(_set.stream));
    auto start = std::chrono::high_resolution_clock::now();
    make_clover<<<_set.gridDim, _set.blockDim, 0, _set.stream>>>(
        gauge, clover, _set.device_lat_xyzt, parity);
    checkCudaErrors(cudaStreamSynchronize(_set.stream));
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
    checkCudaErrors(cudaStreamSynchronize(_set.stream));
    auto start = std::chrono::high_resolution_clock::now();
    inverse_clover<<<_set.gridDim, _set.blockDim, 0, _set.stream>>>(
        clover, _set.device_lat_xyzt);
    checkCudaErrors(cudaStreamSynchronize(_set.stream));
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
    checkCudaErrors(cudaStreamSynchronize(_set.stream));
    auto start = std::chrono::high_resolution_clock::now();
    give_clover<<<_set.gridDim, _set.blockDim, 0, _set.stream>>>(
        clover, fermion_out, _set.device_lat_xyzt);
    checkCudaErrors(cudaStreamSynchronize(_set.stream));
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
  // free
  checkCudaErrors(cudaFreeAsync(clover, _set.stream));
  checkCudaErrors(cudaStreamSynchronize(_set.stream));
  _set.end();
}
#endif
