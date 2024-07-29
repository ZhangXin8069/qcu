#include "../../include/qcu.h"
#ifdef NCCL_WILSON_DSLASH

void ncclDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                   QcuParam *param, int parity, QcuParam *grid) {
  // define for nccl_wilson_dslash
  LatticeSet _set;
  _set.give(param->lattice_size, grid->lattice_size);
  _set.init();
  LatticeWilsonDslash _dslash;
  _dslash.give(&_set);
  _dslash.run_test(fermion_out, fermion_in, gauge, parity);
  _set.end();
}
#endif