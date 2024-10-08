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
#endif
