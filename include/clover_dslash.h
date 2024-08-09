#ifndef _CLOVER_DSLASH_H
#define _CLOVER_DSLASH_H
#pragma once
__global__ void make_clover(void *device_U, void *device_clover,
                            void *device_xyztsc,
                            const int device_parity);

__global__ void inverse_clover(void *device_clover, void *device_xyztsc);

__global__ void give_clover(void *device_clover, void *device_dest,
                            void *device_xyztsc);
#endif