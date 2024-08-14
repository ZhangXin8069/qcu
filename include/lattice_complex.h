#ifndef _LATTICE_COMPLEX_H
#define _LATTICE_COMPLEX_H

#include "./include.h"
using data_type = cuDoubleComplex;

struct LatticeComplex {
  // double2 _data;
  // double &real = ((double *)&_data)[0];
  // double &imag = ((double *)&_data)[1];
  double real;
  double imag;
  __host__ __device__ LatticeComplex(double _real = 0.0, double _imag = 0.0) {
    real = _real;
    imag = _imag;
  }
  __forceinline__ __host__ __device__ LatticeComplex &
  operator=(const LatticeComplex &other) {
    real = other.real;
    imag = other.imag;
    return *this;
  }
  __host__ __device__ LatticeComplex &operator=(const double &other) {
    real = other;
    imag = 0;
    return *this;
  }
  __host__ __device__ LatticeComplex
  operator+(const LatticeComplex &other) const {
    return LatticeComplex(real + other.real, imag + other.imag);
  }
  __host__ __device__ LatticeComplex
  operator-(const LatticeComplex &other) const {
    return LatticeComplex(real - other.real, imag - other.imag);
  }
  __host__ __device__ LatticeComplex
  operator*(const LatticeComplex &other) const {
    return LatticeComplex(real * other.real - imag * other.imag,
                          real * other.imag + imag * other.real);
  }
  __host__ __device__ LatticeComplex operator*(const double &other) const {
    return LatticeComplex(real * other, imag * other);
  }
  __host__ __device__ LatticeComplex
  operator/(const LatticeComplex &other) const {
    double denom = other.real * other.real + other.imag * other.imag;
    return LatticeComplex((real * other.real + imag * other.imag) / denom,
                          (imag * other.real - real * other.imag) / denom);
  }
  __host__ __device__ LatticeComplex operator/(const double &other) const {
    return LatticeComplex(real / other, imag / other);
  }
  __host__ __device__ LatticeComplex operator-() const {
    return LatticeComplex(-real, -imag);
  }
  __host__ __device__ bool operator==(const LatticeComplex &other) const {
    return (real == other.real && imag == other.imag);
  }
  __host__ __device__ bool operator!=(const LatticeComplex &other) const {
    return !(*this == other);
  }
  __host__ __device__ LatticeComplex &operator+=(const LatticeComplex &other) {
    real = real + other.real;
    imag = imag + other.imag;
    return *this;
  }
  __host__ __device__ LatticeComplex &operator-=(const LatticeComplex &other) {
    real = real - other.real;
    imag = imag - other.imag;
    return *this;
  }
  __host__ __device__ LatticeComplex &operator*=(const LatticeComplex &other) {
    real = real * other.real - imag * other.imag;
    imag = real * other.imag + imag * other.real;
    return *this;
  }
  __host__ __device__ LatticeComplex &operator*=(const double &other) {
    real = real * other;
    imag = imag * other;
    return *this;
  }
  __host__ __device__ LatticeComplex &operator/=(const LatticeComplex &other) {
    double denom = other.real * other.real + other.imag * other.imag;
    real = (real * other.real + imag * other.imag) / denom;
    imag = (imag * other.real - real * other.imag) / denom;
    return *this;
  }
  __host__ __device__ LatticeComplex &operator/=(const double &other) {
    real = real / other;
    imag = imag / other;
    return *this;
  }
  __host__ __device__ LatticeComplex conj() const {
    return LatticeComplex(real, -imag);
  }
  __host__ __device__ double norm2() const {
    return sqrt(real * real + imag * imag);
  }
};

#endif