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
  _cg.run_test(gauge);
  _set.end();
  _cg.end();
}
#endif