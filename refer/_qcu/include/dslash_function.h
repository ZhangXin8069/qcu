#ifndef _DSLASH_FUNCTION_H
#define _DSLASH_FUNCTION_H

#include "include.h"

__device__ void F_uv(Complex *U_u, Complex *U_v, int U_u_up, int U_v_up, int U_u_down, int U_v_down, int par_ch, int par_no, Complex *F);
__device__ void inv_33(Complex *a, Complex *a_inv);//3*3矩阵求逆
__device__ void inv_66(Complex *a, Complex *a_inv);//3*3矩阵求逆,a_inv必须是零矩阵
__global__ void dslash(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity);
__global__ void dslash_inner(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity) ;
__global__ void dslash_clover_inner(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity);
__global__ void dslash_border_revc(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_recive, bool if_border) ;
__global__ void dslash_side_revc(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_recive, bool if_border);
__global__ void output_zero( Complex* d_fermi_out);
__global__ void dslash_tborder_rec(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_recive);
__global__ void dslash_xborder_rec(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_recive);
__global__ void dslash_yborder_rec(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_recive);
__global__ void dslash_zborder_rec(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_recive);
__global__ void dslash_tborder(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_send);
__global__ void dslash_xborder(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_send);
__global__ void dslash_yborder(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_send);
__global__ void dslash_zborder(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_send);
__global__ void dot_gpu_fermi_1(Complex* vector_1, Complex* vector_2, Complex* dot_buf, int Lt, const int Lz, const int Ly, const int Lx);
__global__ void dot_gpu_fermi_2(Complex* dot_buf, int Lt, const int Lz, const int Ly, const int Lx, Complex *dot_result);
__global__ void add_gpu_fermi(Complex *fermi_a, Complex *fermi_b, Complex *fermi_c, Complex kappa);
__global__ void add_gpu_fermi_dslash(Complex *fermi_a, Complex *fermi_b, Complex kappa);
__global__ void add_gpu_fermi_2(Complex *fermi_a, Complex *fermi_b, Complex *fermi_c, Complex *fermi_d, Complex  kappa,  Complex  kappa_2);
__global__ void fermi_copy(Complex *fermi_a, Complex *fermi_b);
__global__ void fermi_copy_2(Complex_2 *fermi_a, Complex_2 *fermi_b);


#endif