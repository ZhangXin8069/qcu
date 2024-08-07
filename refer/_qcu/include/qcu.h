#ifndef _QCU_H
#define _QCU_H
#pragma optimize(5)
#pragma once


#define Ls 4
#define Lc 3

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



/*
用于表示U的的位置的指针的宏定义
*/
#define U_t     d_gauge + pos * Lcc + 6 * Ltzyxcc
#define U_z     d_gauge + pos * Lcc + 4 * Ltzyxcc
#define U_y     d_gauge + pos * Lcc + 2 * Ltzyxcc
#define U_x     d_gauge + pos * Lcc
#define U_t_up     Lzyxcc - (t == Lt-1) * Ltzyxcc
#define U_t_down   - Lzyxcc + (t == 0) * Ltzyxcc
#define U_z_up     Lyxcc - (z == Lz-1) * Lzyxcc
#define U_z_down   - Lyxcc + (z == 0) * Lzyxcc
#define U_y_up     Lxcc - (y == Ly-1) * Lyxcc
#define U_y_down   - Lxcc + (y == 0) * Lyxcc
#define U_x_up     (d_parity != (t+z+y)%2)*( Lcc - (x == Lx-1) * Lxcc)
#define U_x_down   (d_parity == (t+z+y)%2)*( -Lcc + (x == 0) * Lxcc) 
#define U_parity_change      (d_parity == 0 ) * Ltzyxcc
#define U_parity_nochange    (d_parity ) * Ltzyxcc

#define for_ijk_mat for(int i = 0; i < Lc; i++)for(int j = 0; j < Lc; j++)for(int k = 0; k < Lc; k++)


#ifdef __cplusplus
}
#endif

#endif