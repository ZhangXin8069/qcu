#ifndef _BISTABCG_H
#define _BISTABCG_H
#include "./lattice_complex.h"
#include "./cg.h"
__global__ void bistabcg_give_1beta(void *device_vals);
__global__ void bistabcg_give_1rho_prev(void *device_vals);
__global__ void bistabcg_give_1alpha(void *device_vals);
__global__ void bistabcg_give_1omega(void *device_vals);
__global__ void bistabcg_give_rr(void *device_r, void *device_b__o,
                                 void *device_r_tilde, void *device_vals);
__global__ void bistabcg_give_p(void *device_p, void *device_r, void *device_v,
                                void *device_vals);
__global__ void bistabcg_give_s(void *device_s, void *device_r, void *device_v,
                                void *device_vals);
__global__ void bistabcg_give_x_o(void *device_x_o, void *device_p,
                                  void *device_s, void *device_vals);
__global__ void bistabcg_give_r(void *device_r, void *device_s, void *device_tt,
                                void *device_vals);
#endif