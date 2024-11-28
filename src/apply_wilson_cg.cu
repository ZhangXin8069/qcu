#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;

void applyCgQcu(void *fermion_out, void *fermion_in, void *gauge,
                QcuParam *param, QcuParam *grid)
{
  // define for apply_wilson_cg
  LatticeSet<double> _set;
  _set.give(param->lattice_size, grid->lattice_size);
  _set.init();
  dptzyxcc2ccdptzyx<double>(gauge, &_set);
  ptzyxsc2psctzyx<double>(fermion_in, &_set);
  ptzyxsc2psctzyx<double>(fermion_out, &_set);
  LatticeCg<double> _cg;
  _cg.give(&_set);
  _cg.init(fermion_out, fermion_in, gauge);
  _cg.run_test();
  _cg.end();
  ccdptzyx2dptzyxcc<double>(gauge, &_set);
  psctzyx2ptzyxsc<double>(fermion_in, &_set);
  psctzyx2ptzyxsc<double>(fermion_out, &_set);
  _set.end();
}