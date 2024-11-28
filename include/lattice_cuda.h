#ifndef _LATTICE_CUDA_H
#define _LATTICE_CUDA_H
#include "./include.h"
#include "./lattice_set.h"
namespace qcu
{
  template <typename T>
  __global__ void give_random_vals(void *device_random_vals, unsigned long seed);
  template <typename T>
  __global__ void give_custom_vals(void *device_custom_vals, double real,
                                   double imag);
  template <typename T>
  __global__ void give_1zero(void *device_vals, const int vals_index);
  template <typename T>
  __global__ void give_1one(void *device_vals, const int vals_index);
  template <typename T>
  __global__ void give_1custom(void *device_vals, const int vals_index,
                               double real, double imag);
  template <typename T>
  __global__ void _tzyxsc2sctzyx(void *device_fermi, void *device__fermi,
                                 int lat_4dim);
  template <typename T>
  __global__ void _sctzyx2tzyxsc(void *device_fermi, void *device__fermi,
                                 int lat_4dim);
  template <typename T>
  void tzyxsc2sctzyx(void *fermion, LatticeSet<T> *set_ptr);
  template <typename T>
  void sctzyx2tzyxsc(void *fermion, LatticeSet<T> *set_ptr);
  template <typename T>
  __global__ void _dptzyxcc2ccdptzyx(void *device_gauge, void *device__gauge,
                                     int lat_4dim);
  template <typename T>
  __global__ void _ccdptzyx2dptzyxcc(void *device_gauge, void *device__gauge,
                                     int lat_4dim);
  template <typename T>
  void dptzyxcc2ccdptzyx(void *gauge, LatticeSet<T> *set_ptr);
  template <typename T>
  void ccdptzyx2dptzyxcc(void *gauge, LatticeSet<T> *set_ptr);
  template <typename T>
  __global__ void _ptzyxsc2psctzyx(void *device_fermi, void *device__fermi,
                                   int lat_4dim);
  template <typename T>
  __global__ void _psctzyx2ptzyxsc(void *device_fermi, void *device__fermi,
                                   int lat_4dim);
  template <typename T>
  void ptzyxsc2psctzyx(void *fermion, LatticeSet<T> *set_ptr);
  template <typename T>
  void psctzyx2ptzyxsc(void *fermion, LatticeSet<T> *set_ptr);
  template <typename T>
  __global__ void give_debug_u(void *device_U, void *device_params);
}
#endif