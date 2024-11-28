#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;

void applyDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                    QcuParam *param, int parity, QcuParam *grid)
{
  // define for apply_wilson_dslash
  LatticeSet<double> _set;
  _set.give(param->lattice_size, grid->lattice_size, parity);
  _set.init();
  dptzyxcc2ccdptzyx<double>(gauge, &_set);
  tzyxsc2sctzyx<double>(fermion_in, &_set);
  tzyxsc2sctzyx<double>(fermion_out, &_set);
  LatticeWilsonDslash<double> _wilson_dslash;
  _wilson_dslash.give(&_set);
  _wilson_dslash.run_test(fermion_out, fermion_in, gauge);
  ccdptzyx2dptzyxcc<double>(gauge, &_set);
  sctzyx2tzyxsc<double>(fermion_in, &_set);
  sctzyx2tzyxsc<double>(fermion_out, &_set);
  _set.end();
}