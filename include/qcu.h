#ifndef _QCU_H
#define _QCU_H
#pragma once
#include "./define.h"
#include "./include.h"
#include "./lattice_cg.h"
#include "./lattice_bistabcg.h"
#include "./lattice_clover_dslash.h"
#include "./lattice_complex.h"
#include "./lattice_cuda.h"
#include "./lattice_set.h"
#include "./lattice_wilson_dslash.h"
typedef struct QcuParam_s
{
  int lattice_size[4];
} QcuParam;
template <typename T>
void testDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                   QcuParam *param, int parity);
template <typename T>
void applyDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                    QcuParam *param, int parity, QcuParam *grid);
template <typename T>
void testCloverDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                         QcuParam *param, int parity);
template <typename T>
void applyCloverDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                          QcuParam *param, int parity, QcuParam *grid);
template <typename T>
void applyBistabCgQcu(void *fermion_out, void *fermion_in, void *gauge,
                      QcuParam *param, QcuParam *grid);
template <typename T>
void applyCgQcu(void *fermion_out, void *fermion_in, void *gauge,
                QcuParam *param, QcuParam *grid);
#endif