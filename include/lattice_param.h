#ifndef _LATTICE_PARAM_H
#define _LATTICE_PARAM_H
#include "./qcu.h"

struct LatticeParam {
  int lat_size[_DIM_];
  LatticeParam(int *param_lat_size) {
    lat_size[_X_] = param_lat_size[_X_];
    lat_size[_Y_] = param_lat_size[_Y_];
    lat_size[_Z_] = param_lat_size[_Z_];
    lat_size[_T_] = param_lat_size[_T_];
  }
  LatticeParam() {}
  void __LAT_EXAMPLE_() {
    lat_size[_X_] = _LAT_EXAMPLE_;
    lat_size[_Y_] = _LAT_EXAMPLE_;
    lat_size[_Z_] = _LAT_EXAMPLE_;
    lat_size[_T_] = _LAT_EXAMPLE_;
  }
  void __GRID_EXAMPLE_() {
    lat_size[_X_] = _GRID_EXAMPLE_;
    lat_size[_Y_] = _GRID_EXAMPLE_;
    lat_size[_Z_] = _GRID_EXAMPLE_;
    lat_size[_T_] = _GRID_EXAMPLE_;
  }
};

#endif