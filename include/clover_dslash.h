#ifndef _CLOVER_DSLASH_H
#define _CLOVER_DSLASH_H
__global__ void make_clover(void *device_U, void *device_clover,
                            void *device_lat_xyzt, const int device_parity);

__global__ void inverse_clover(void *device_clover, void *device_lat_xyzt);

__global__ void give_clover(void *device_clover, void *device_dest,
                            void *device_lat_xyzt);
__global__ void make_clover_inside(void *device_U, void *device_clover,
                                   void *device_lat_xyzt, int device_parity);
__global__ void pick_up_u(void *device_U, void *device_lat_xyzt,
                          void **device_u_1dim_send_vec,
                          void **device_u_2dim_send_vec, int device_parity);


#endif