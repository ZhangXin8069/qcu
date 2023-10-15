#pragma optimize(5)
#include "./define.h"
#include <cmath>

struct LatticePoint {
}

struct LatticePoint {

  __device__ LatticePoint(const double &real = 0.0, const double &imag = 0.0)
      : real(real), imag(imag) {}
  __device__ LatticePoint &operator=(const LatticePoint &other) {
    real = other.real;
    imag = other.imag;
    return *this;
  }
  __device__ LatticePoint &operator=(const double &other) {
    real = other;
    imag = 0;
    return *this;
  }
  __device__ LatticePoint operator+(const LatticePoint &other) const {
    return LatticePoint(real + other.real, imag + other.imag);
  }
  __device__ LatticePoint operator-(const LatticePoint &other) const {
    return LatticePoint(real - other.real, imag - other.imag);
  }
  __device__ LatticePoint operator*(const LatticePoint &other) const {
    return LatticePoint(real * other.real - imag * other.imag,
                        real * other.imag + imag * other.real);
  }
  __device__ LatticePoint operator*(const double &other) const {
    return LatticePoint(real * other, imag * other);
  }
  __device__ LatticePoint operator/(const LatticePoint &other) const {
    double denom = other.real * other.real + other.imag * other.imag;
    return LatticePoint((real * other.real + imag * other.imag) / denom,
                        (imag * other.real - real * other.imag) / denom);
  }
  __device__ LatticePoint operator/(const double &other) const {
    return LatticePoint(real / other, imag / other);
  }
  __device__ LatticePoint operator-() const {
    return LatticePoint(-real, -imag);
  }
  __device__ bool operator==(const LatticePoint &other) const {
    return (real == other.real && imag == other.imag);
  }
  __device__ bool operator!=(const LatticePoint &other) const {
    return !(*this == other);
  }
  __device__ LatticePoint &operator+=(const LatticePoint &other) {
    real = real + other.real;
    imag = imag + other.imag;
    return *this;
  }
  __device__ LatticePoint &operator-=(const LatticePoint &other) {
    real = real - other.real;
    imag = imag - other.imag;
    return *this;
  }
  __device__ LatticePoint &operator*=(const LatticePoint &other) {
    real = real * other.real - imag * other.imag;
    imag = real * other.imag + imag * other.real;
    return *this;
  }
  __device__ LatticePoint &operator*=(const double &other) {
    real = real * other;
    imag = imag * other;
    return *this;
  }
  __device__ LatticePoint &operator/=(const LatticePoint &other) {
    double denom = other.real * other.real + other.imag * other.imag;
    real = (real * other.real + imag * other.imag) / denom;
    imag = (imag * other.real - real * other.imag) / denom;
    return *this;
  }
  __device__ LatticePoint &operator/=(const double &other) {
    real = real / other;
    imag = imag / other;
    return *this;
  }
  __device__ LatticePoint conj() const { return LatticePoint(real, -imag); }
  __device__ double norm2() const { return sqrt(real * real + imag * imag); }
};