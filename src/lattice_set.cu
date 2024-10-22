#include "../include/qcu.h"
#ifdef LATTICE_SET
__global__ void give_param(void *device_param, int vals_index, int val) {
  int *param = static_cast<int *>(device_param);
  param[vals_index] = val;
}
#endif