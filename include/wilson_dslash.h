#ifndef _WILSON_DSLASH_H
#define _WILSON_DSLASH_H
/*
b_send
*/
__global__ void wilson_dslash_b_x_send(void *device_src, void *device_xyztsc,
                                       const int device_parity,
                                       void *device_b_x_send_vec);
__global__ void wilson_dslash_b_y_send(void *device_src, void *device_xyztsc,
                                       const int device_parity,
                                       void *device_b_y_send_vec);
__global__ void wilson_dslash_b_z_send(void *device_src, void *device_xyztsc,
                                       const int device_parity,
                                       void *device_b_z_send_vec);
__global__ void wilson_dslash_b_t_send(void *device_src, void *device_xyztsc,
                                       const int device_parity,
                                       void *device_b_t_send_vec);
/*
f_send
*/
__global__ void wilson_dslash_f_x_send(void *device_U, void *device_src,
                                       void *device_xyztsc,
                                       const int device_parity,
                                       void *device_f_x_send_vec);
__global__ void wilson_dslash_f_y_send(void *device_U, void *device_src,
                                       void *device_xyztsc,
                                       const int device_parity,
                                       void *device_f_y_send_vec);
__global__ void wilson_dslash_f_z_send(void *device_U, void *device_src,
                                       void *device_xyztsc,
                                       const int device_parity,
                                       void *device_f_z_send_vec);
__global__ void wilson_dslash_f_t_send(void *device_U, void *device_src,
                                       void *device_xyztsc,
                                       const int device_parity,
                                       void *device_f_t_send_vec);
/*
compute
*/
__global__ void wilson_dslash_x_compute(void *device_U, void *device_src,
                                        void *device_dest, void *device_xyztsc,
                                        const int device_parity);
__global__ void wilson_dslash_y_compute(void *device_U, void *device_src,
                                        void *device_dest, void *device_xyztsc,
                                        const int device_parity);
__global__ void wilson_dslash_z_compute(void *device_U, void *device_src,
                                        void *device_dest, void *device_xyztsc,
                                        const int device_parity);
__global__ void wilson_dslash_t_compute(void *device_U, void *device_src,
                                        void *device_dest, void *device_xyztsc,
                                        const int device_parity);
/*
b_recv
*/
__global__ void wilson_dslash_b_x_recv(void *device_dest, void *device_xyztsc,
                                       const int device_parity,
                                       void *device_b_x_recv_vec);
__global__ void wilson_dslash_b_y_recv(void *device_dest, void *device_xyztsc,
                                       const int device_parity,
                                       void *device_b_y_recv_vec);
__global__ void wilson_dslash_b_z_recv(void *device_dest, void *device_xyztsc,
                                       const int device_parity,
                                       void *device_b_z_recv_vec);
__global__ void wilson_dslash_b_t_recv(void *device_dest, void *device_xyztsc,
                                       const int device_parity,
                                       void *device_b_t_recv_vec);
/*
f_recv
*/
__global__ void wilson_dslash_f_x_recv(void *device_U, void *device_dest,
                                       void *device_xyztsc,
                                       const int device_parity,
                                       void *device_f_x_recv_vec);
__global__ void wilson_dslash_f_y_recv(void *device_U, void *device_dest,
                                       void *device_xyztsc,
                                       const int device_parity,
                                       void *device_f_y_recv_vec);
__global__ void wilson_dslash_f_z_recv(void *device_U, void *device_dest,
                                       void *device_xyztsc,
                                       const int device_parity,
                                       void *device_f_z_recv_vec);
__global__ void wilson_dslash_f_t_recv(void *device_U, void *device_dest,
                                       void *device_xyztsc,
                                       const int device_parity,
                                       void *device_f_t_recv_vec);
/*
single wilson dslash
*/
__global__ void wilson_dslash(void *device_U, void *device_src,
                              void *device_dest, void *device_xyztsc,
                              const int device_parity);
#endif