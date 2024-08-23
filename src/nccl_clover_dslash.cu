#include "../include/qcu.h"
#ifdef NCCL_CLOVER_DSLASH
void ncclDslashCloverQcu(void *fermion_out, void *fermion_in, void *gauge,
                         QcuParam *param, int parity, QcuParam *grid) {
  // define for nccl_clover_dslash
  LatticeSet _set;
  _set.give(param->lattice_size, grid->lattice_size);
  _set.init();
   if(_set.node_rank == 0) _set._print(); // test
  dptzyxcc2ccdptzyx(gauge, &_set);
  tzyxsc2sctzyx(fermion_in, &_set);
  tzyxsc2sctzyx(fermion_out, &_set);
  LatticeWilsonDslash _wilson_dslash;
  LatticeCloverDslash _clover_dslash;
  _wilson_dslash.give(&_set);
  _clover_dslash.give(&_set);
  _clover_dslash.init();
  {
    // wilson dslash
    _wilson_dslash.run_test(fermion_out, fermion_in, gauge, parity);
  }
  {
    // make clover
    _clover_dslash.make(gauge, parity);
  }
  {
    // inverse clover
    _clover_dslash.inverse();
  }
  {
    // give clover
    _clover_dslash.give(fermion_out);
  }
  ccdptzyx2dptzyxcc(gauge, &_set);
  sctzyx2tzyxsc(fermion_in, &_set);
  sctzyx2tzyxsc(fermion_out, &_set);
  _clover_dslash.end();
  _set.end();
}
#endif