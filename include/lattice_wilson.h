#ifndef _LATTICE_WILSON_H
#define _LATTICE_WILSON_H

#include "./qcu.h"
__global__ void wilson_dslash(void *device_U, void *device_src,
                              void *device_dest, int device_lat_x,
                              const int device_lat_y, const int device_lat_z,
                              const int device_lat_t, const int device_parity);

__global__ void wilson_dslash_clear_dest(void *device_dest, int device_lat_x,
                                         const int device_lat_y,
                                         const int device_lat_z);

__global__ void
wilson_dslash_x_send(void *device_U, void *device_src, void *device_dest,
                     int device_lat_x, const int device_lat_y,
                     const int device_lat_z, const int device_lat_t,
                     const int device_parity, void *device_b_x_send_vec,
                     void *device_f_x_send_vec);

__global__ void
wilson_dslash_x_recv(void *device_U, void *device_dest, int device_lat_x,
                     const int device_lat_y, const int device_lat_z,
                     const int device_lat_t, const int device_parity,
                     void *device_b_x_recv_vec, void *device_f_x_recv_vec);

__global__ void
wilson_dslash_y_send(void *device_U, void *device_src, void *device_dest,
                     int device_lat_x, const int device_lat_y,
                     const int device_lat_z, const int device_lat_t,
                     const int device_parity, void *device_b_y_send_vec,
                     void *device_f_y_send_vec);

__global__ void
wilson_dslash_y_recv(void *device_U, void *device_dest, int device_lat_x,
                     const int device_lat_y, const int device_lat_z,
                     const int device_lat_t, const int device_parity,
                     void *device_b_y_recv_vec, void *device_f_y_recv_vec);

__global__ void
wilson_dslash_z_send(void *device_U, void *device_src, void *device_dest,
                     int device_lat_x, const int device_lat_y,
                     const int device_lat_z, const int device_lat_t,
                     const int device_parity, void *device_b_z_send_vec,
                     void *device_f_z_send_vec);

__global__ void
wilson_dslash_z_recv(void *device_U, void *device_dest, int device_lat_x,
                     const int device_lat_y, const int device_lat_z,
                     const int device_lat_t, const int device_parity,
                     void *device_b_z_recv_vec, void *device_f_z_recv_vec);

__global__ void
wilson_dslash_t_send(void *device_U, void *device_src, void *device_dest,
                     int device_lat_x, const int device_lat_y,
                     const int device_lat_z, const int device_lat_t,
                     const int device_parity, void *device_b_t_send_vec,
                     void *device_f_t_send_vec);

__global__ void
wilson_dslash_t_recv(void *device_U, void *device_dest, int device_lat_x,
                     const int device_lat_y, const int device_lat_z,
                     const int device_lat_t, const int device_parity,
                     void *device_b_t_recv_vec, void *device_f_t_recv_vec);

__global__ void test_wilson_dslash(void *device_U, void *device_src,
                                   void *device_dest, int device_lat_x,
                                   const int device_lat_y,
                                   const int device_lat_z,
                                   const int device_lat_t,
                                   const int device_parity);

__global__ void wilson_bistabcg_give_b_e(void *device_b_e, void *device_ans_e,
                                         void *device_latt_tmp0, double kappa);

__global__ void wilson_bistabcg_give_b_o(void *device_b_o, void *device_ans_o,
                                         void *device_latt_tmp1, double kappa);

__global__ void wilson_bistabcg_give_b__0(void *device_b__o, void *device_b_o,
                                          void *device_latt_tmp0, double kappa);

__global__ void wilson_bistabcg_give_dest_o(void *device_dest_o,
                                            void *device_src_o,
                                            void *device_latt_tmp1,
                                            double kappa);

__global__ void wilson_bistabcg_give_rr(void *device_r, void *device_b__o,
                                        void *device_r_tilde);

__global__ void wilson_bistabcg_give_p(void *device_p, void *device_r,
                                       void *device_v, LatticeComplex omega,
                                       LatticeComplex beta);

__global__ void wilson_bistabcg_give_s(void *device_s, void *device_r,
                                       void *device_v, LatticeComplex alpha);

__global__ void wilson_bistabcg_give_x_o(void *device_x_o, void *device_p,
                                         void *device_s, LatticeComplex alpha,
                                         LatticeComplex omega);

__global__ void wilson_bistabcg_give_r(void *device_r, void *device_s,
                                       void *device_tt, LatticeComplex omega);
#endif