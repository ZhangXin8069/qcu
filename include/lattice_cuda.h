#ifndef _LATTICE_CUDA_H
#define _LATTICE_CUDA_H


__global__ void give_random_value(void *device_random_value,
                                  unsigned long seed);

__global__ void give_custom_value(void *device_custom_value, double real,
                                  double imag);

#endif