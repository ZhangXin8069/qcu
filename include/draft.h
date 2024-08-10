#ifndef _DRAFT_H
#define _DRAFT_H
#pragma once
// clang-format off
#include "./define.h"
// clang-format on
__global__ void part_dot(void *device_vec0, void *device_vec1,
                         void *device_dot_vec);

__global__ void part_cut(void *device_vec0, void *device_vec1,
                         void *device_dot_vec);

void perf_part_reduce(void *device_src_vec, void *device_dest_val,
                     void *device_tmp_vec, int size, cudaStream_t stream);

void part_reduce(void *device_src_vec, void *device_dest_val,
                     void *device_tmp_vec, int size, cudaStream_t stream);
                     
#endif