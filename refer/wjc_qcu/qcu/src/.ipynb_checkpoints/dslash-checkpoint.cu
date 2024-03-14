#include <cstdio>
#include <cmath>
#include <assert.h>
#include <chrono>
#include <mpi.h>
#include "qcu.h"
#include <cuda_runtime.h>
#include "qcu_complex.cuh"
#include "qcu_dslash.cuh"
#include "qcu_macro.cuh"
#include "qcu_complex_computation.cuh"
#include "qcu_point.cuh"
#include "qcu_communicator.cuh"
#include "qcu_clover_dslash.cuh"

#define qcuPrint() { \
    printf("function %s line %d...\n", __FUNCTION__, __LINE__); \
}




void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity) {

  DslashParam dslash_param(fermion_in, fermion_out, gauge, param, parity);
  // MpiWilsonDslash dslash_solver(dslash_param);
  CloverDslash dslash_solver(dslash_param);
  dslash_solver.calculateDslash(1);
}
/*
void callCloverDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag) {
  DslashParam dslash_param(fermion_in, fermion_out, gauge, param, parity);
  CloverDslash dslash_solver(dslash_param);
  dslash_solver.calculateDslash(invert_flag);
}
*/
