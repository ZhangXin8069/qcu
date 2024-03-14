#ifndef _COMPLEX_H
#define _COMPLEX_H
#pragma optimize(5)
#include "./qcu.h"

// Complex number class
class Complex {
public:
  // Data members
  double real;
  double imag;

  // Constructors

  // Default constructor
  __host__ __device__ Complex() : real(0.0), imag(0.0) {}

  // Constructor with real and imaginary parts
  __host__ __device__ Complex(double r, double i) : real(r), imag(i) {}

  // Accessors
  __host__ __device__ double _real() const { return real; }
  __host__ __device__ double _imag() const { return imag; }
  __host__ __device__ void _real(double r) { real = r; }
  __host__ __device__ void _imag(double i) { imag = i; }

  // Destructor
  __host__ __device__ ~Complex() {}
  
  // Arithmetic operators
  __host__ __device__ Complex operator+(const Complex &rhs) const {
    return {real + rhs.real, imag + rhs.imag};
  }
  __host__ __device__ Complex operator+(double rhs) const {
    return {real + rhs, imag};
  }
  __host__ __device__ Complex operator-(const Complex &rhs) const {
    return {real - rhs.real, imag - rhs.imag};
  }
  __host__ __device__ Complex operator-(double rhs) const {
    return {real - rhs, imag};
  }
  __host__ __device__ Complex operator*(const Complex &rhs) const {
    return {real * rhs.real - imag * rhs.imag,
            real * rhs.imag + imag * rhs.real};
  }
  __host__ __device__ Complex operator*(double rhs) const {
    return {real * rhs, imag * rhs};
  }
  __host__ __device__ Complex operator/(const Complex &rhs) const {
    double denominator = rhs.real * rhs.real + rhs.imag * rhs.imag;
    return {(real * rhs.real + imag * rhs.imag) / denominator,
            (imag * rhs.real - real * rhs.imag) / denominator};
  }
  __host__ __device__ Complex operator/(double rhs) const {
    return {real / rhs, imag / rhs};
  }

  // Assignment operators
  __host__ __device__ Complex &operator=(const Complex &rhs) {
    real = rhs.real;
    imag = rhs.imag;
    return *this;
  }
  __host__ __device__ Complex &operator=(const double &rhs) {
    real = rhs;
    imag = 0;
    return *this;
  }
  __host__ __device__ Complex &operator+=(const Complex &rhs) {
    real += rhs.real;
    imag += rhs.imag;
    return *this;
  }
  __host__ __device__ Complex &operator+=(double rhs) {
    real += rhs;
    return *this;
  }
  __host__ __device__ Complex &operator-=(const Complex &rhs) {
    real -= rhs.real;
    imag -= rhs.imag;
    return *this;
  }
  __host__ __device__ Complex &operator-=(double rhs) {
    real -= rhs;
    return *this;
  }
  __host__ __device__ Complex &operator*=(const Complex &rhs) {
    real = real * rhs.real - imag * rhs.imag;
    imag = real * rhs.imag + imag * rhs.real;
    return *this;
  }
  __host__ __device__ Complex &operator*=(double rhs) {
    real *= rhs;
    imag *= rhs;
    return *this;
  }
  __host__ __device__ Complex &operator/=(const Complex &rhs) {
    double denominator = rhs.real * rhs.real + rhs.imag * rhs.imag;
    real = (real * rhs.real + imag * rhs.imag) / denominator;
    imag = (imag * rhs.real - real * rhs.imag) / denominator;
    return *this;
  }
  __host__ __device__ Complex &operator/=(double rhs) {
    real /= rhs;
    imag /= rhs;
    return *this;
  }

  // Unary minus
  __host__ __device__ Complex operator-() const { return {-real, -imag}; }

  // Conjugate
  __host__ __device__ Complex conj() const { return {real, -imag}; }

  // Multiplication with i
  __host__ __device__ Complex mul_i() const {
    return { -imag, real};
  }

  // Multiplication with ii
  __host__ __device__ Complex mul_ii() const {
    return { -imag, -real};
  }

  // Multiplication with iii
  __host__ __device__ Complex mul_iii() const {
    return { imag, -real};
  }

  // String representation
  __host__ std::string to_string() const {
    return std::to_string(real) + " + " + std::to_string(imag) + "i";
  }
};

#endif