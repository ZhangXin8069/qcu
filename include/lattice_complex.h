#ifndef _LATTICE_COMPLEX_H
#define _LATTICE_COMPLEX_H

#include "./include.h"
using data_type = cuDoubleComplex;

struct LatticeComplex {
  double2 _data;
  __host__ __device__ __inline__ LatticeComplex(const double &_real = 0.0,
                                                const double &_imag = 0.0) {
    _data.x = _real;
    _data.y = _imag;
  }
  __host__ __device__ __inline__ LatticeComplex &
  operator=(const LatticeComplex &other) {
    _data.x = other._data.x;
    _data.y = other._data.y;
    return *this;
  }
  __host__ __device__ __inline__ LatticeComplex &
  operator=(const double &other) {
    _data.x = other;
    _data.y = 0;
    return *this;
  }
  __host__ __device__ __inline__ LatticeComplex
  operator+(const LatticeComplex &other) const {
    return LatticeComplex(_data.x + other._data.x, _data.y + other._data.y);
  }
  __host__ __device__ __inline__ LatticeComplex
  operator-(const LatticeComplex &other) const {
    return LatticeComplex(_data.x - other._data.x, _data.y - other._data.y);
  }
  __host__ __device__ __inline__ LatticeComplex
  operator*(const LatticeComplex &other) const {
    return LatticeComplex(_data.x * other._data.x - _data.y * other._data.y,
                          _data.x * other._data.y + _data.y * other._data.x);
  }
  __host__ __device__ __inline__ LatticeComplex
  operator*(const double &other) const {
    return LatticeComplex(_data.x * other, _data.y * other);
  }
  __host__ __device__ __inline__ LatticeComplex
  operator/(const LatticeComplex &other) const {
    double denom =
        other._data.x * other._data.x + other._data.y * other._data.y;
    return LatticeComplex(
        (_data.x * other._data.x + _data.y * other._data.y) / denom,
        (_data.y * other._data.x - _data.x * other._data.y) / denom);
  }
  __host__ __device__ __inline__ LatticeComplex
  operator/(const double &other) const {
    return LatticeComplex(_data.x / other, _data.y / other);
  }
  __host__ __device__ __inline__ LatticeComplex operator-() const {
    return LatticeComplex(-_data.x, -_data.y);
  }
  __host__ __device__ __inline__ bool
  operator==(const LatticeComplex &other) const {
    return (_data.x == other._data.x && _data.y == other._data.y);
  }
  __host__ __device__ __inline__ bool
  operator!=(const LatticeComplex &other) const {
    return !(*this == other);
  }
  __host__ __device__ __inline__ LatticeComplex &
  operator+=(const LatticeComplex &other) {
    _data.x = _data.x + other._data.x;
    _data.y = _data.y + other._data.y;
    return *this;
  }
  __host__ __device__ __inline__ LatticeComplex &
  operator-=(const LatticeComplex &other) {
    _data.x = _data.x - other._data.x;
    _data.y = _data.y - other._data.y;
    return *this;
  }
  __host__ __device__ __inline__ LatticeComplex &
  operator*=(const LatticeComplex &other) {
    _data.x = _data.x * other._data.x - _data.y * other._data.y;
    _data.y = _data.x * other._data.y + _data.y * other._data.x;
    return *this;
  }
  __host__ __device__ __inline__ LatticeComplex &
  operator*=(const double &other) {
    _data.x = _data.x * other;
    _data.y = _data.y * other;
    return *this;
  }
  __host__ __device__ __inline__ LatticeComplex &
  operator/=(const LatticeComplex &other) {
    double denom =
        other._data.x * other._data.x + other._data.y * other._data.y;
    _data.x = (_data.x * other._data.x + _data.y * other._data.y) / denom;
    _data.y = (_data.y * other._data.x - _data.x * other._data.y) / denom;
    return *this;
  }
  __host__ __device__ __inline__ LatticeComplex &
  operator/=(const double &other) {
    _data.x = _data.x / other;
    _data.y = _data.y / other;
    return *this;
  }
  __host__ __device__ __inline__ LatticeComplex conj() const {
    return LatticeComplex(_data.x, -_data.y);
  }
  __host__ __device__ __inline__ double norm2() const {
    return sqrt(_data.x * _data.x + _data.y * _data.y);
  }
  friend std::ostream &operator<<(std::ostream &output, const LatticeComplex &_) {
    output << "(" << _._data.x << "," << _._data.y << "i"
           << ")";
    return output;
  }
};

#endif