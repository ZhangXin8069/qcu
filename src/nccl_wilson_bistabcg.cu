#include "../include/qcu.h"
#include "lattice_set.h"
#ifdef NCCL_WILSON_BISTABCG

void ncclBistabCgQcu(void *fermion_out, void *fermion_in, void *gauge,
                     QcuParam *param, QcuParam *grid) {
  // define for nccl_wilson_bistabcg
  LatticeSet _set;
  _set.give(param->lattice_size, grid->lattice_size);
  _set.init();
  tzyxdcc2dcctzyx(gauge, &_set);
  tzyxsc2sctzyx(fermion_in, &_set);
  tzyxsc2sctzyx(fermion_out, &_set);
  LatticeBistabcg _bistabcg;
  _bistabcg.give(&_set);
  _bistabcg.init(fermion_out, fermion_in, gauge);
  // _bistabcg.run();
  _bistabcg.end();
  // dcctzyx2tzyxdcc(gauge, &_set);
  // sctzyx2tzyxsc(fermion_in, &_set);
  // sctzyx2tzyxsc(fermion_out, &_set);
  give_custom_value<<<_set.gridDim, _set.blockDim, 0, _set.stream>>>(fermion_in, 666.0, 999.0);
  give_custom_value<<<_set.gridDim, _set.blockDim, 0, _set.stream>>>(fermion_out, 999.0, 666.0);
  _set._print();
  _set.end();
}
#endif