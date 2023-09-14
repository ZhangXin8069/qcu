#include "./define.h"
#include "./lattice_complex.h"
#include <cuda.h>
#include <mpi.h>
__global__ void wilson_dslash(void *device_U, void *device_src,
                              void *device_dest, int device_lat_x,
                              const int device_lat_y, const int device_lat_z,
                              const int device_lat_t, const int device_parity);

__global__ void mpi_wilson_dslash(void *device_U, void *device_src,
                              void *device_dest, int device_lat_x,
                              const int device_lat_y, const int device_lat_z,
                              const int device_lat_t, const int device_parity,
                              const int device_node_rank, int device_grid_x,
                              const int device_grid_y, const int device_grid_z,
                              const int device_grid_t);

__global__ void make_clover(void *device_U, void *device_clover,
                            int device_lat_x, const int device_lat_y,
                            const int device_lat_z, const int device_lat_t,
                            const int device_parity);

__global__ void inverse_clover(void *device_clover, int device_lat_x,
                               const int device_lat_y, const int device_lat_z);

__global__ void give_clover(void *device_clover, void *device_dest,
                            int device_lat_x, const int device_lat_y,
                            const int device_lat_z);