#ifndef _QCU_H
#define _QCU_H
#pragma optimize(5)
#pragma once
#include "./include.h"
#include "./define.h"
#include "./complex.h"
#include "./complex_vector.h"
#include "./qcu_cuda.h"
#include "./qcu_mpi.h"
#include "./lattice_complex.h"
#include "./lattice_param.h"
#include "./lattice_point.h"
#include "./lattice_gamma.h"
#include "./lattice_fermi.h"
#include "./lattice_propagator.h"
#include "./lattice_gauge.h"
#include "./lattice_clover.h"
#include "./lattice_overlap.h"
#include "./lattice_wilson.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QcuParam_s {
  int lattice_size[4];
} QcuParam;

void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity);
void mpiDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, QcuParam *grid);
void mpiCgQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, QcuParam *grid);
void testDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity);

#ifdef __cplusplus
}
#endif

#endif