#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
#ifdef NCCL_WILSON_CG
void ncclCgQcu(void *fermion_out, void *fermion_in, void *gauge,
                     QcuParam *param, QcuParam *grid) {
  // define for nccl_wilson_cg
  LatticeSet _set;
  _set.give(param->lattice_size, grid->lattice_size);
  _set.init();
  dptzyxcc2ccdptzyx(gauge, &_set);
  ptzyxsc2psctzyx(fermion_in, &_set);
  ptzyxsc2psctzyx(fermion_out, &_set);
  // LatticeBistabCg _cg;
  // printf("this version just is for test, in fact used bistabcg function, not cg function.\n");
  LatticeCg _cg;
  _cg.give(&_set);
  _cg.init(fermion_out, fermion_in, gauge);
  _cg.run();
  _cg.end();
  ccdptzyx2dptzyxcc(gauge, &_set);
  psctzyx2ptzyxsc(fermion_in, &_set);
  psctzyx2ptzyxsc(fermion_out, &_set);
  _set.end();
}
#endif