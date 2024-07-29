#ifndef _CLOVER_DSLASH_H
#define _CLOVER_DSLASH_H

__global__ void test_wilson_dslash(void *device_U, void *device_src,
                                   void *device_dest, int device_lat_x,
                                   const int device_lat_y,
                                   const int device_lat_z,
                                   const int device_lat_t,
                                   const int device_parity);

__global__ void make_clover(void *device_U, void *device_clover,
                            int device_lat_x, const int device_lat_y,
                            const int device_lat_z, const int device_lat_t,
                            const int device_parity);

__global__ void inverse_clover(void *device_clover, int device_lat_x,
                               const int device_lat_y, const int device_lat_z);

__global__ void give_clover(void *device_clover, void *device_dest,
                            int device_lat_x, const int device_lat_y,
                            const int device_lat_z);
#endif