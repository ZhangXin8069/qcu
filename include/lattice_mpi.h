#ifndef _LATTICE_MPI_H
#define _LATTICE_MPI_H

__global__ void bistabcg_part_dot(void *device_dots,
                                         void *device_val0, void *device_val1);

__global__ void bistabcg_part_cut(void *device_vec0,
                                         void *device_val0, void *device_val1);

#endif
