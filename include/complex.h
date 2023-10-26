#ifndef _COMPLEX_H
#define _COMPLEX_H
#pragma optimize(5)
#include "./qcu_cuda.h"

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

  // Arithmetic operators
  __host__ __device__ Complex operator+(const Complex &rhs) const {
    return {real + rhs.real, imag + rhs.imag};
  }

  __host__ __device__ Complex operator-(const Complex &rhs) const {
    return {real - rhs.real, imag - rhs.imag};
  }

  __host__ __device__ Complex operator*(const Complex &rhs) const {
    return {real * rhs.real - imag * rhs.imag,
            real * rhs.imag + imag * rhs.real};
  }

  __host__ __device__ Complex operator/(const Complex &rhs) const {
    double denominator = rhs.real * rhs.real + rhs.imag * rhs.imag;
    return {(real * rhs.real + imag * rhs.imag) / denominator,
            (imag * rhs.real - real * rhs.imag) / denominator};
  }

  // Unary minus
  __host__ __device__ Complex operator-() const { return {-real, -imag}; }

  // Conjugate
  __host__ __device__ Complex conjugate() const { return {real, -imag}; }

  // Magnitude
  double magnitude() const { return std::sqrt(real * real + imag * imag); }

  // Argument
  double argument() const { return std::atan2(imag, real); }

  // Normalize
  __host__ __device__ Complex &normalize() {
    double magnitude = this->magnitude();
    real /= magnitude;
    imag /= magnitude;
    return *this;
  }

  // Assignment operator
  __host__ __device__ Complex &operator=(const Complex &rhs) {
    if (this == &rhs) {
      return *this;
    }

    real = rhs.real;
    imag = rhs.imag;

    return *this;
  }

  // Arithmetic operators with double, int, and float

  // Addition
  __host__ __device__ Complex operator+(double rhs) const {
    return {real + rhs, imag};
  }

  __host__ __device__ Complex operator+(int rhs) const {
    return {real + rhs, imag};
  }

  __host__ __device__ Complex operator+(float rhs) const {
    return {real + rhs, imag};
  }

  // Subtraction
  __host__ __device__ Complex operator-(double rhs) const {
    return {real - rhs, imag};
  }

  __host__ __device__ Complex operator-(int rhs) const {
    return {real - rhs, imag};
  }

  __host__ __device__ Complex operator-(float rhs) const {
    return {real - rhs, imag};
  }

  // Multiplication
  __host__ __device__ Complex operator*(double rhs) const {
    return {real * rhs, imag * rhs};
  }

  __host__ __device__ Complex operator*(int rhs) const {
    return {real * rhs, imag * rhs};
  }

  __host__ __device__ Complex operator*(float rhs) const {
    return {real * rhs, imag * rhs};
  }

  // Division
  __host__ __device__ Complex operator/(double rhs) const {
    return {real / rhs, imag / rhs};
  }

  __host__ __device__ Complex operator/(int rhs) const {
    return {real / rhs, imag / rhs};
  }

  __host__ __device__ Complex operator/(float rhs) const {
    return {real / rhs, imag / rhs};
  }

  // Assignment operators
  __host__ __device__ Complex &operator+=(const Complex &rhs) {
    real += rhs.real;
    imag += rhs.imag;
    return *this;
  }

  __host__ __device__ Complex &operator-=(const Complex &rhs) {
    real -= rhs.real;
    imag -= rhs.imag;
    return *this;
  }

  __host__ __device__ Complex &operator*=(const Complex &rhs) {
    real = real * rhs.real - imag * rhs.imag;
    imag = real * rhs.imag + imag * rhs.real;
    return *this;
  }

  __host__ __device__ Complex &operator/=(const Complex &rhs) {
    double denominator = rhs.real * rhs.real + rhs.imag * rhs.imag;
    real = (real * rhs.real + imag * rhs.imag) / denominator;
    imag = (imag * rhs.real - real * rhs.imag) / denominator;
    return *this;
  }

  // Unary add assignment operator
  __host__ __device__ Complex &operator+=(double rhs) {
    real += rhs;
    return *this;
  }

  __host__ __device__ Complex &operator+=(int rhs) {
    real += rhs;
    return *this;
  }

  __host__ __device__ Complex &operator+=(float rhs) {
    real += rhs;
    return *this;
  }

  // Unary minus assignment operator
  __host__ __device__ Complex &operator-=(double rhs) {
    real -= rhs;
    return *this;
  }

  __host__ __device__ Complex &operator-=(int rhs) {
    real -= rhs;
    return *this;
  }

  __host__ __device__ Complex &operator-=(float rhs) {
    real -= rhs;
    return *this;
  }

  // Multiplication assignment operator
  __host__ __device__ Complex &operator*=(double rhs) {
    real *= rhs;
    imag *= rhs;
    return *this;
  }

  __host__ __device__ Complex &operator*=(int rhs) {
    real *= rhs;
    imag *= rhs;
    return *this;
  }

  __host__ __device__ Complex &operator*=(float rhs) {
    real *= rhs;
    imag *= rhs;
    return *this;
  }

  // Division assignment operator
  __host__ __device__ Complex &operator/=(double rhs) {
    real /= rhs;
    imag /= rhs;
    return *this;
  }

  __host__ __device__ Complex &operator/=(int rhs) {
    real /= rhs;
    imag /= rhs;
    return *this;
  }

  __host__ __device__ Complex &operator/=(float rhs) {
    real /= rhs;
    imag /= rhs;
    return *this;
  }

  // Accessors
  double _real() const { return real; }
  double _imag() const { return imag; }

  void _real(double r) { real = r; }
  void _imag(double i) { imag = i; }

  // String representation
  std::string to_string() const {
    return std::to_string(real) + " + " + std::to_string(imag) + "i";
  }
};
#endif