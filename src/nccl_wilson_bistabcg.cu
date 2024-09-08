// clang-format off
#include "../include/qcu.h"
#include "lattice_complex.h"
#include "lattice_set.h"
#ifdef NCCL_WILSON_BISTABCG
void ncclBistabCgQcu(void *fermion_out, void *fermion_in, void *gauge,
                     QcuParam *param, QcuParam *grid) {
  // define for nccl_wilson_bistabcg
  LatticeSet _set;
  _set.give(param->lattice_size, grid->lattice_size);
  _set.init();
  // if(_set.node_rank == 0) _set._print(); // test
  dptzyxcc2ccdptzyx(gauge, &_set);
  ptzyxsc2psctzyx(fermion_in, &_set);
  ptzyxsc2psctzyx(fermion_out, &_set);
  LatticeBistabcg _bistabcg;
  _bistabcg.give(&_set);
  _bistabcg.init(fermion_out, fermion_in, gauge);
  _bistabcg.run();
  _bistabcg.end();
  ccdptzyx2dptzyxcc(gauge, &_set);
  psctzyx2ptzyxsc(fermion_in, &_set);
  psctzyx2ptzyxsc(fermion_out, &_set);
  _set.end();
  printf("###%d\n",int(sizeof(double2)));
  printf("###%d\n",int(sizeof(LatticeComplex)));
}
#endif