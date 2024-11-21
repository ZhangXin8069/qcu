#pragma once
#ifdef __cplusplus
extern "C" {
#endif
typedef struct QcuParam_s {
  int lattice_size[4];
} QcuParam;
void testDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
               QcuParam *param, int parity);
void testCloverDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                     QcuParam *param, int parity);
void mpiDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                  QcuParam *param, int parity, QcuParam *grid);
void mpiBistabCgQcu(void *gauge, QcuParam *param, QcuParam *grid);
void applyDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                   QcuParam *param, int parity, QcuParam *grid);
void applyBistabCgQcu(void *fermion_out, void *fermion_in, void *gauge,
                     QcuParam *param, QcuParam *grid);
void applyCloverDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                         QcuParam *param, int parity, QcuParam *grid);
#ifdef __cplusplus
}
#endif
