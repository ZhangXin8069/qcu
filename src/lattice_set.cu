#include "../include/qcu.h"
#pragma optimize(5)
namespace qcu
{
  template <typename T = double>
template <typename T = double>
  __global__ void give_param(void *device_param, int vals_index, int val)
  {
    int *param = static_cast<int *>(device_param);
    param[vals_index] = val;
  }
}