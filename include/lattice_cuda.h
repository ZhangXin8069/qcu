#ifndef _LATTICE_CUDA_H
#define _LATTICE_CUDA_H

__global__ void give_random_value(void *device_random_value,
                                  unsigned long seed);

__global__ void give_custom_value(void *device_custom_value, double real,
                                  double imag);

__global__ void give_1zero(void *device_vals, const int vals_index);

__global__ void give_1one(void *device_vals, const int vals_index);

__global__ void fermion_dot(void *device_vec0, void *device_vec1,
                            void *device_vals, const int vals_index);

__global__ void fermion_diff(void *device_vec0, void *device_vec1,
                             void *device_vals, const int vals_index);

__global__ void bistabcg_add(void *device_vec0, void *device_vec1,
                             void *device_vals, const int vals_index);

__global__ void fermion_subt(void *device_vec0, void *device_vec1,
                             void *device_vals, const int vals_index);

__global__ void fermion_mult(void *device_vec0, void *device_vec1,
                             void *device_vals, const int vals_index);

__global__ void fermion_divi(void *device_vec0, void *device_vec1,
                             void *device_vals, const int vals_index);

#endif