#ifndef _LATTICE_CUDA_H
#define _LATTICE_CUDA_H
#include "./include.h"
__global__ void give_fermi_rand(void *device_random_value,
                                  unsigned long seed);
__global__ void give_fermi_val(void *device_custom_value, double real,
                                  double imag);
__global__ void give_1zero(void *device_vals, const int vals_index);
__global__ void give_1one(void *device_vals, const int vals_index);
__global__ void part_dot(void *device_vec0, void *device_vec1,
                         void *device_dot_vec);
__global__ void part_cut(void *device_vec0, void *device_vec1,
                         void *device_dot_vec);
void perf_part_reduce(void *device_src_vec, void *device_dest_val,
                     void *device_tmp_vec, int size, cudaStream_t stream);
void part_reduce(void *device_src_vec, void *device_dest_val,
                     void *device_tmp_vec, int size, cudaStream_t stream);
#endif