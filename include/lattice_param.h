#ifndef _LATTICE_PARAM_H
#define _LATTICE_PARAM_H
#pragma optimize(5)
#include "./qcu.h"

// Lattice param class
class LatticeParam {
public:
  // Data members, sort as x, y, z, t
  int _lat_size[LAT_D];
  int _grid_size[LAT_D];
  int _parity;

  // Constructors

  // Default constructor
  __host__ __device__ LatticeParam() {}

  // Constructor with data
  __host__ __device__ LatticeParam(int lat_x, int lat_y, int lat_z, int lat_t,
                                   int grid_x, int grid_y, int grid_z,
                                   int grid_t, int parity) {
    _lat_size[X] = lat_x;
    _lat_size[Y] = lat_y;
    _lat_size[Z] = lat_z;
    _lat_size[T] = lat_t;
    _grid_size[X] = grid_x;
    _grid_size[Y] = grid_y;
    _grid_size[Z] = grid_z;
    _grid_size[T] = grid_t;
    _parity = parity;
  }

  // Constructor with other
  __host__ __device__ LatticeParam(const LatticeParam &other){
    _parity = other._parity;
    for (int i = 0; i < LAT_D; ++i) {
      _lat_size[i] = other._lat_size[i];
      _grid_size[i] = other._grid_size[i];
    }
  }

  // Constructor with pointer to data
  __host__ __device__ LatticeParam(int *lat_ptr, int *grid_ptr, int parity) {
    _parity = parity;
    for (int i = 0; i < LAT_D; ++i) {
      _lat_size[i] = lat_ptr[i];
      _grid_size[i] = grid_ptr[i];
    }
  }

  // Destructor
  // __host__ ~LatticeParam() {}
  // __device__ ~LatticeParam() {}

  // Assignment operators
  __host__ __device__ LatticeParam &operator=(const LatticeParam &other) {
    _parity = other._parity;
    for (int i = 0; i < LAT_D; ++i) {
      _lat_size[i] = other._lat_size[i];
      _grid_size[i] = other._grid_size[i];
    }
    return *this;
  }

  // Get lattice
  __host__ __device__ int lat(int dim) const { return _lat_size[dim]; }

  // Get lattice size
  __host__ __device__ int lat_size() const {
    return _lat_size[X] * _lat_size[Y] * _lat_size[Z] * _lat_size[T];
  }

  // Get grid
  __host__ __device__ int grid(int dim) const { return _grid_size[dim]; }

  // Get grid size
  __host__ __device__ int grid_size() const {
    return _grid_size[X] * _grid_size[Y] * _grid_size[Z] * _grid_size[T];
  }

  // String representation
  __host__ std::string lat_to_string() const {
    std::string result;
    for (int i = 0; i < LAT_D; ++i) {
      result += std::to_string(_lat_size[i]) + " ";
    }
    return result;
  }
  __host__ std::string grid_to_string() const {
    std::string result;
    for (int i = 0; i < LAT_D; ++i) {
      result += std::to_string(_grid_size[i]) + " ";
    }
    return result;
  }
  __host__ std::string to_string() const {
    std::string result;
    result = lat_to_string() + grid_to_string() + "#" + std::to_string(_parity) + "#";
    return result;
  }
};

#endif