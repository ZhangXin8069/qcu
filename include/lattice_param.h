#pragma optimize(5)
#include "./define.h"
#include <cmath>

struct LatticeParam {

  int papram[12];       // grid,lat,index
  int fact_index[2];    // U,scr(dest)
  int change_papram[2]; // parity,eo
  __device__ LatticeParam() : real(real), imag(imag) {}
  __device__ LatticeParam &operator=(const LatticeParam &other) {
    real = other.real;
    imag = other.imag;
    return *this;
  }
  __device__ LatticeParam &operator=(const double &other) {
    real = other;
    imag = 0;
    return *this;
  }
  __device__ LatticeParam operator+(const LatticeParam &other) const {
    return LatticeParam(real + other.real, imag + other.imag);
  }
  __device__ LatticeParam operator-(const LatticeParam &other) const {
    return LatticeParam(real - other.real, imag - other.imag);
  }
  __device__ LatticeParam operator*(const LatticeParam &other) const {
    return LatticeParam(real * other.real - imag * other.imag,
                        real * other.imag + imag * other.real);
  }
  __device__ LatticeParam operator*(const double &other) const {
    return LatticeParam(real * other, imag * other);
  }
  __device__ LatticeParam operator/(const LatticeParam &other) const {
    double denom = other.real * other.real + other.imag * other.imag;
    return LatticeParam((real * other.real + imag * other.imag) / denom,
                        (imag * other.real - real * other.imag) / denom);
  }
  __device__ LatticeParam operator/(const double &other) const {
    return LatticeParam(real / other, imag / other);
  }
  __device__ LatticeParam operator-() const {
    return LatticeParam(-real, -imag);
  }
  __device__ bool operator==(const LatticeParam &other) const {
    return (real == other.real && imag == other.imag);
  }
  __device__ bool operator!=(const LatticeParam &other) const {
    return !(*this == other);
  }
  __device__ LatticeParam &operator+=(const LatticeParam &other) {
    real = real + other.real;
    imag = imag + other.imag;
    return *this;
  }
  __device__ LatticeParam &operator-=(const LatticeParam &other) {
    real = real - other.real;
    imag = imag - other.imag;
    return *this;
  }
  __device__ LatticeParam &operator*=(const LatticeParam &other) {
    real = real * other.real - imag * other.imag;
    imag = real * other.imag + imag * other.real;
    return *this;
  }
  __device__ LatticeParam &operator*=(const double &other) {
    real = real * other;
    imag = imag * other;
    return *this;
  }
  __device__ LatticeParam &operator/=(const LatticeParam &other) {
    double denom = other.real * other.real + other.imag * other.imag;
    real = (real * other.real + imag * other.imag) / denom;
    imag = (imag * other.real - real * other.imag) / denom;
    return *this;
  }
  __device__ LatticeParam &operator/=(const double &other) {
    real = real / other;
    imag = imag / other;
    return *this;
  }
  __device__ LatticeParam conj() const { return LatticeParam(real, -imag); }
  __device__ double norm2() const { return sqrt(real * real + imag * imag); }
};