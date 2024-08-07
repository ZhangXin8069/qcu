#ifndef _QCU_MPI_H
#define _QCU_MPI_H

#include "./qcu.h"

__global__ void wilson_bistabcg_part_dot(void *device_dot_tmp,
                                         void *device_val0, void *device_val1);

__global__ void wilson_bistabcg_part_cut(void *device_latt_tmp0,
                                         void *device_val0, void *device_val1);

#endif
