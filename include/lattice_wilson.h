#ifndef _LATTICE_WILSON_H
#define _LATTICE_WILSON_H
#pragma optimize(5)
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

#endif