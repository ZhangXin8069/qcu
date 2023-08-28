#include <cmath>
struct LatticeComplex {
  double real;
  double imag;
  __forceinline__ __device__ LatticeComplex(const double &real = 0.0,
                                            const double &imag = 0.0)
      : real(real), imag(imag) {}
  __forceinline__ __device__ LatticeComplex &
  operator=(const LatticeComplex &other) {
    real = other.real;
    imag = other.imag;
    return *this;
  }
  __forceinline__ __device__ LatticeComplex &operator=(const double &other) {
    real = other;
    imag = 0;
    return *this;
  }
  __forceinline__ __device__ LatticeComplex
  operator+(const LatticeComplex &other) const {
    return LatticeComplex(real + other.real, imag + other.imag);
  }
  __forceinline__ __device__ LatticeComplex
  operator-(const LatticeComplex &other) const {
    return LatticeComplex(real - other.real, imag - other.imag);
  }
  __forceinline__ __device__ LatticeComplex
  operator*(const LatticeComplex &other) const {
    return LatticeComplex(real * other.real - imag * other.imag,
                          real * other.imag + imag * other.real);
  }
  __forceinline__ __device__ LatticeComplex
  operator*(const double &other) const {
    return LatticeComplex(real * other, imag * other);
  }
  __forceinline__ __device__ LatticeComplex
  operator/(const LatticeComplex &other) const {
    double denom = other.real * other.real + other.imag * other.imag;
    return LatticeComplex((real * other.real + imag * other.imag) / denom,
                          (imag * other.real - real * other.imag) / denom);
  }
  __forceinline__ __device__ LatticeComplex
  operator/(const double &other) const {
    return LatticeComplex(real / other, imag / other);
  }
  __forceinline__ __device__ LatticeComplex operator-() const {
    return LatticeComplex(-real, -imag);
  }
  __forceinline__ __device__ bool
  operator==(const LatticeComplex &other) const {
    return (real == other.real && imag == other.imag);
  }
  __forceinline__ __device__ bool
  operator!=(const LatticeComplex &other) const {
    return !(*this == other);
  }
  __forceinline__ __device__ LatticeComplex &
  operator+=(const LatticeComplex &other) {
    real = real + other.real;
    imag = imag + other.imag;
    return *this;
  }
  __forceinline__ __device__ LatticeComplex &
  operator-=(const LatticeComplex &other) {
    real = real - other.real;
    imag = imag - other.imag;
    return *this;
  }
  __forceinline__ __device__ LatticeComplex &
  operator*=(const LatticeComplex &other) {
    real = real * other.real - imag * other.imag;
    imag = real * other.imag + imag * other.real;
    return *this;
  }
  __forceinline__ __device__ LatticeComplex &operator*=(const double &other) {
    real = real * other;
    imag = imag * other;
    return *this;
  }
  __forceinline__ __device__ LatticeComplex &
  operator/=(const LatticeComplex &other) {
    double denom = other.real * other.real + other.imag * other.imag;
    real = (real * other.real + imag * other.imag) / denom;
    imag = (imag * other.real - real * other.imag) / denom;
    return *this;
  }
  __forceinline__ __device__ LatticeComplex &operator/=(const double &other) {
    real = real / other;
    imag = imag / other;
    return *this;
  }
  __forceinline__ __device__ LatticeComplex conj() const {
    return LatticeComplex(real, -imag);
  }
  __forceinline__ __device__ double norm2() const {
    return sqrt(real * real + imag * imag);
  }
};