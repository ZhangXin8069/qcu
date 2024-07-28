#ifndef _LATTICE_CLOVER_H
#define _LATTICE_CLOVER_H
#pragma optimize(5)
#include "./qcu.h"

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