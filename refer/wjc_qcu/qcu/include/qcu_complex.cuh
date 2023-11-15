#pragma once

#include <cstdio>
// #ifdef __cplusplus
// extern "C" {
// #endif

class Complex {
private:
  double real_;
  double imag_;

public:
  __device__ __host__ Complex(double real, double imag) : real_(real), imag_(imag) {}
  Complex() = default;
  __device__ __host__ Complex(const Complex &complex) : real_(complex.real_), imag_(complex.imag_) {}
  __device__ __host__ double norm2() { return sqrt(real_ * real_ + imag_ * imag_); }
  __device__ __host__ void setImag(double imag) { imag_ = imag; }
  __device__ __host__ void setReal(double real) { real_ = real; }
  __device__ __host__ double real() const { return real_; }
  __device__ __host__ double imag() const { return imag_; }

  __device__ __host__ Complex &operator=(const Complex &complex)
  {
    real_ = complex.real_;
    imag_ = complex.imag_;
    return *this;
  }
  __device__ __host__ Complex &operator=(double rhs)
  {
    real_ = rhs;
    imag_ = 0;
    return *this;
  }
  __device__ __host__ Complex operator+(const Complex &complex) const
  {
    return Complex(real_ + complex.real_, imag_ + complex.imag_);
  }
  __device__ __host__ Complex operator-(const Complex &complex) const
  {
    return Complex(real_ - complex.real_, imag_ - complex.imag_);
  }
  __device__ __host__ Complex operator-() const { return Complex(-real_, -imag_); }
  __device__ __host__ Complex operator*(const Complex &rhs) const
  {
    return Complex(real_ * rhs.real_ - imag_ * rhs.imag_, real_ * rhs.imag_ + imag_ * rhs.real_);
  }
  __device__ __host__ Complex operator*(const double &rhs) const
  {
    return Complex(real_ * rhs, imag_ * rhs);
  }
  // __device__ __host__ Complex &operator*=(const Complex &rhs) // TO modify
  // {
  //   real_ = real_ * rhs.real_ - imag_ * rhs.imag_;
  //   imag_ = real_ * rhs.imag_ + imag_ * rhs.real_;
  //   return *this;
  // }
  __device__ __host__ Complex &operator*=(const double &rhs)
  {
    real_ = real_ * rhs;
    imag_ = imag_ * rhs;
    return *this;
  }
  __device__ __host__ Complex operator/(const double &rhs) { return Complex(real_ / rhs, imag_ / rhs); }
  __device__ __host__ Complex operator/(const Complex &rhs) const {
    return (*this * rhs.conj()) / (rhs.real()*rhs.real() + rhs.imag()*rhs.imag());
  }
  __device__ __host__ Complex& operator/=(const Complex &rhs) {
    double new_real = (real_*rhs.real() + imag_*rhs.imag()) / (rhs.real()*rhs.real() + rhs.imag()*rhs.imag());
    double new_imag = (rhs.real()*imag_ - real_*rhs.imag()) / (rhs.real()*rhs.real() + rhs.imag()*rhs.imag());
    real_ = new_real;
    imag_ = new_imag;
    return *this;
  }
  __device__ __host__ Complex &operator+=(const Complex &rhs)
  {
    real_ += rhs.real_;
    imag_ += rhs.imag_;
    return *this;
  }

  __device__ __host__ Complex &operator-=(const Complex &rhs)
  {
    real_ -= rhs.real_;
    imag_ -= rhs.imag_;
    return *this;
  }

  __device__ __host__ Complex &clear2Zero()
  {
    real_ = 0;
    imag_ = 0;
    return *this;
  }


  __device__ __host__ Complex multipy_i()
  {
    return Complex(-imag_, real_);
  }
  __device__ __host__ Complex multipy_minus_i()
  {
    return Complex(imag_, -real_);
  }
  __device__ __host__ Complex &self_multipy_i()
  {
    // return Complex(-imag_, real_);
    double temp = real_;
    real_ = -imag_;
    imag_ = temp;
    return *this;
  }
  __device__ __host__ Complex &self_multipy_minus_i()
  {
    // return Complex(imag_, -real_);
    double temp = -real_;
    real_ = imag_;
    imag_ = temp;
    return *this;
  }

  __device__ __host__ Complex conj() const { return Complex(real_, -imag_); }
  __device__ __host__ bool operator==(const Complex &rhs) { return real_ == rhs.real_ && imag_ == rhs.imag_; }
  __device__ __host__ bool operator!=(const Complex &rhs) { return real_ != rhs.real_ || imag_ != rhs.imag_; }
  __device__ __host__ void output() const { printf("(%lf + %lfi)", real_, imag_); }
  // friend __device__ __host__ Complex& operator*(int lhs, const Complex& rhs);
};
// __device__ __host__ Complex operator*(int lhs, const Complex& rhs) {
//   return Complex(lhs * rhs.real(), lhs * rhs.imag());
// }
// #ifdef __cplusplus
// }
// #endif
