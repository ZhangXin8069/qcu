#include "../../include/qcu.h"
#ifdef NCCL_WILSON_BISTABCG

void ncclBistabCgQcu(void *gauge, QcuParam *param, QcuParam *grid) {
  // define for nccl_wilson_bistabcg
  LatticeSet _set;
  _set.give(param->lattice_size, grid->lattice_size);
  _set.init();
  LatticeBistabcg _cg;
  _cg.give(&_set);
  _cg.init(gauge);
  auto start = std::chrono::high_resolution_clock::now();
  // nccl wilson bistabcg
  _cg.run_test(gauge);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  _set.err = cudaGetLastError();
  checkCudaErrors(_set.err);
  printf("nccl wilson bistabcg total time: (without malloc free memcpy) :%.9lf "
         "sec\n",
         double(duration) / 1e9);
  _set.end();
  _cg.end();
}
#endif