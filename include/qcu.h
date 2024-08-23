#ifndef _QCU_H
#define _QCU_H
#pragma once
#include "./define.h"
#include "./include.h"
#include "./lattice_bistabcg.h"
#include "./lattice_clover_dslash.h"
#include "./lattice_complex.h"
#include "./lattice_cuda.h"
#include "./lattice_set.h"
#include "./lattice_wilson_dslash.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct QcuParam_s {
  int lattice_size[4];
} QcuParam;
void dslashQcu(void *fermion_out, void *fermion_in, void *gauge,
               QcuParam *param, int parity);
void dslashCloverQcu(void *fermion_out, void *fermion_in, void *gauge,
                     QcuParam *param, int parity);
void mpiDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                  QcuParam *param, int parity, QcuParam *grid);
void mpiBistabCgQcu(void *gauge, QcuParam *param, QcuParam *grid);
void ncclDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                   QcuParam *param, int parity, QcuParam *grid);
void ncclBistabCgQcu(void *fermion_out, void *fermion_in, void *gauge,
                     QcuParam *param, QcuParam *grid);
void ncclDslashCloverQcu(void *fermion_out, void *fermion_in, void *gauge,
                         QcuParam *param, int parity, QcuParam *grid);
#ifdef __cplusplus
}
#endif
#endif