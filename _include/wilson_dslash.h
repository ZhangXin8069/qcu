#ifndef __wilson_dslash_H
#define __wilson_dslash_H

__global__ void wilson_dslash(void *device_U, void *device_src,
                              void *device_dest, void *device_xyztsc,
                              const int device_parity);

__global__ void _wilson_dslash_clear_dest(void *device_dest,
                                         void *device_xyztsc);

#endif