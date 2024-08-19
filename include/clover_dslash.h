#ifndef _CLOVER_DSLASH_H
#define _CLOVER_DSLASH_H
__global__ void make_clover(void *device_U, void *device_clover,
                            void *device_lat_xyzt,
                            const int device_parity);

__global__ void inverse_clover(void *device_clover, void *device_lat_xyzt);

__global__ void give_clover(void *device_clover, void *device_dest,
                            void *device_lat_xyzt);
#endif