#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QcuParam_s {
  int lattice_size[4];
} QcuParam;

void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity);
void dslashCloverQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity);
void mpiDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, QcuParam *grid);
void mpiBistabCgQcu(void *gauge, QcuParam *param, QcuParam *grid);
void ncclDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, QcuParam *grid);
void ncclBistabCgQcu(void *gauge, QcuParam *param, QcuParam *grid);

#ifdef __cplusplus
}
#endif
