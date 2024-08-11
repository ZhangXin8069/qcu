#include "../include/qcu.h"
#ifdef NCCL_WILSON_BISTABCG

void ncclBistabCgQcu(void *fermion_out, void *fermion_in, void *gauge,
                     QcuParam *param, QcuParam *grid) {
  // define for nccl_wilson_bistabcg
  LatticeSet _set;
  _set.give(param->lattice_size, grid->lattice_size);
  _set.init();
  LatticeBistabcg _cg;
  _cg.give(&_set);
  _cg.init(fermion_out, fermion_in, gauge);
  _cg.run();
  _cg.end();
  _set.end();
}
#endif