#include "../include/qcu.h"
#ifdef _QCU_NCCL_WILSON_DSLASH_
void ncclDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                   QcuParam *param, int parity, QcuParam *grid) {
  // define for nccl_wilson_dslash
  LatticeSet _set;
  _set.give(param->lattice_size, grid->lattice_size, parity);
  _set.init();
  dptzyxcc2ccdptzyx(gauge, &_set);
  tzyxsc2sctzyx(fermion_in, &_set);
  tzyxsc2sctzyx(fermion_out, &_set);
  LatticeWilsonDslash _wilson_dslash;
  _wilson_dslash.give(&_set);
  _wilson_dslash.run_test(fermion_out, fermion_in, gauge);
  ccdptzyx2dptzyxcc(gauge, &_set);
  sctzyx2tzyxsc(fermion_in, &_set);
  sctzyx2tzyxsc(fermion_out, &_set);
  _set.end();
}
#endif