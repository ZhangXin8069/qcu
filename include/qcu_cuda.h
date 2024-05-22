#ifndef _QCU_CUDA_H
#define _QCU_CUDA_H
#pragma optimize(5)
#include "./qcu.h"

__global__ void give_random_value(void *device_random_value,
                                  unsigned long seed);

__global__ void give_custom_value(void *device_custom_value, double real,
                                  double imag);

#endif