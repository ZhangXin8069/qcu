#ifndef _LATTICE_POINT_H
#define _LATTICE_POINT_H
#pragma optimize(5)
#include "./qcu.h"

// Lattice point class
class LatticePoint {
public:
  // Data members, sort as x, y, z, t
  LatticeParam _param;
  int _lat_index[LAT_D];
  int _eo;
  int _lat_index_;

  // Constructors

  // Default constructor
  __host__ __device__ LatticePoint() {}

  // Constructor with Lattice parameters
  __host__ __device__ LatticePoint(const LatticeParam &param, int blockIdx_x,
                                   int blockDim_x, int threadIdx_x) {
    _param = param;
    int tmp0;
    int tmp1;
    tmp0 = blockIdx_x * blockDim_x + threadIdx_x;
    tmp1 = _param.lat(X) * _param.lat(Y) * _param.lat(Z);
    _lat_index[T] = tmp0 / tmp1;
    tmp0 -= _lat_index[T] * tmp1;
    tmp1 = _param.lat(X) * _param.lat(Y);
    _lat_index[Z] = tmp0 / tmp1;
    tmp0 -= _lat_index[Z] * tmp1;
    _lat_index[Y] = tmp0 / _param.lat(X);
    _eo = get_eo();
    _lat_index_ = get_index();
  }

  // Init with Lattice parameters
  __host__ __device__ void init_param(const LatticeParam &param) {
    _param = param;
  }

  // Init with 3 int
  __host__ __device__ void init_3int(int blockIdx_x, int blockDim_x,
                                     int threadIdx_x) {
    int tmp0;
    int tmp1;
    tmp0 = blockIdx_x * blockDim_x + threadIdx_x;
    tmp1 = _param.lat(X) * _param.lat(Y) * _param.lat(Z);
    _lat_index[T] = tmp0 / tmp1;
    tmp0 -= _lat_index[T] * tmp1;
    tmp1 = _param.lat(X) * _param.lat(Y);
    _lat_index[Z] = tmp0 / tmp1;
    tmp0 -= _lat_index[Z] * tmp1;
    _lat_index[Y] = tmp0 / _param.lat(X);
    _eo = get_eo();
    _lat_index_ = get_index();
  }

  // Constructor with other LatticePoint
  __host__ __device__ LatticePoint(const LatticePoint &other) {
    _param = other._param;
    for (int i = 0; i < LAT_D; ++i) {
      _lat_index[i] = other._lat_index[i];
    }
    _eo = get_eo();
    _lat_index_ = get_index();
  }

  // Destructor
  __device__ __host__ ~LatticePoint() {}

  // Move
  __device__ __host__ LatticePoint move(int Dim, int Ward) {
    LatticePoint result;
    int r = _lat_index[Dim];
    int lat_r = _param.lat(Dim);
    for (int i = 0; i < LAT_D; ++i) {
      result._lat_index[i] = _lat_index[i];
    }
    result._param = _param;
    result._param._parity = (FORWARD - _param._parity) * (Ward != NOWARD) +
                            _param._parity * (Ward == NOWARD);
    if (Ward == BACKWARD) {
      result._lat_index[Dim] +=
          (-1 + (r == 0) * lat_r) *
          ((_eo == _param._parity) * (Dim == X) + (Dim != X));
    }
    if (Ward == FORWARD) {
      result._lat_index[Dim] +=
          (1 - (r == lat_r - 1) * lat_r) *
          ((_eo != _param._parity) * (Dim == X) + (Dim != X));
    }
    result._eo = result.get_eo();
    result._lat_index_ = result.get_index();
    return result;
  }

  // Get lattice point index
  __host__ __device__ int get_index() {
    _lat_index_ =
        _lat_index[0] +
        (_lat_index[1] +
         (_lat_index[2] + _lat_index[3] * _param.lat(2)) * _param.lat(1)) *
            _param.lat(0);
    return _lat_index_;
  }

  // Get lattice point eo
  __host__ __device__ int get_eo() {
    _eo = (_lat_index[Y] + _lat_index[Z] + _lat_index[T]) & 0x01;
    return _eo;
  }

  // String representation
  __host__ std::string index_to_string() const {
    std::string result;
    for (int i = 0; i < LAT_D; ++i) {
      result += std::to_string(_lat_index[i]) + " ";
    }
    return result;
  }
  __host__ std::string to_string() const {
    std::string result;
    result = _param.to_string() + index_to_string() + "#" +
             std::to_string(_eo) + "#" + std::to_string(_lat_index_) + "#";
    return result;
  }
};

#endif