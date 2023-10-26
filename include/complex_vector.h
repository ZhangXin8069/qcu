#ifndef _COMPLEX_VECTOR_H
#define _COMPLEX_VECTOR_H
#pragma optimize(5)
#include "./qcu_cuda.h"

// Complex vector class
class ComplexVector {
public:
  // Constructors

  // Default constructor
  ComplexVector() : _data(nullptr), _size(0) {}

  // Constructor with size
  ComplexVector(int size) : _data(new Complex[size]), _size(size) {}

  // Copy constructor
  ComplexVector(const ComplexVector &other)
      : _data(new Complex[other._size]), _size(other._size) {
    for (int i = 0; i < _size; ++i) {
      _data[i] = other._data[i];
    }
  }

  // Move constructor
  ComplexVector(ComplexVector &&other) noexcept
      : _data(other._data), _size(other._size) {
    other._data = nullptr;
    other._size = 0;
  }

  // Destructor
  ~ComplexVector() { delete[] _data; }

  // Element access
  Complex &operator[](int index) { return _data[index]; }

  const Complex &operator[](int index) const { return _data[index]; }

  // Arithmetic operators

  // Optimized element-wise addition
  // Uses the AVX vectorization instruction set if available
  ComplexVector operator+(const ComplexVector &rhs) const {
#if defined(__AVX__)
    __m256d this_vector = _mm256_load_pd(_data);
    __m256d other_vector = _mm256_load_pd(rhs._data);
    __m256d result_vector = _mm256_add_pd(this_vector, other_vector);
    ComplexVector result(_size);
    _mm256_store_pd(result._data, result_vector);
    return result;
#else
    ComplexVector result(_size);
    for (int i = 0; i < _size; ++i) {
      result[i] = _data[i] + rhs[i];
    }
    return result;
#endif
  }

  // Optimized element-wise subtraction
  // Uses the AVX vectorization instruction set if available
  ComplexVector operator-(const ComplexVector &rhs) const {
#if defined(__AVX__)
    __m256d this_vector = _mm256_load_pd(_data);
    __m256d other_vector = _mm256_load_pd(rhs._data);
    __m256d result_vector = _mm256_sub_pd(this_vector, other_vector);
    ComplexVector result(_size);
    _mm256_store_pd(result._data, result_vector);
    return result;
#else
    ComplexVector result(_size);
    for (int i = 0; i < _size; ++i) {
      result[i] = _data[i] - rhs[i];
    }
    return result;
#endif
  }

  // Optimized element-wise multiplication
  // Uses the AVX vectorization instruction set if available
  ComplexVector operator*(double scale) const {
#if defined(__AVX__)
    __m256d this_vector = _mm256_load_pd(_data);
    __m256d scaled_vector = _mm256_mul_pd(this_vector, _mm256_set1_pd(scale));
    ComplexVector result(_size);
    _mm256_store_pd(result._data, scaled_vector);
    return result;
#else
    ComplexVector result(_size);
    for (int i = 0; i < _size; ++i) {
      result[i] = _data[i] * scale;
    }
    return result;
#endif
  }

  // Arithmetic operators with double, int, and float

  // Addition with double
  ComplexVector operator+(double rhs) const {
    ComplexVector result(_size);
    for (int i = 0; i < _size; ++i) {
      result[i] = _data[i] + rhs;
    }
    return result;
  }

  // Addition with int
  ComplexVector operator+(int rhs) const {
    ComplexVector result(_size);
    for (int i = 0; i < _size; ++i) {
      result[i] = _data[i] + rhs;
    }
    return result;
  }

  // Addition with float
  ComplexVector operator+(float rhs) const {
    ComplexVector result(_size);
    for (int i = 0; i < _size; ++i) {
      result[i] = _data[i] + rhs;
    }
    return result;
  }

  // Subtraction with double
  ComplexVector operator-(double rhs) const {
    ComplexVector result(_size);
    for (int i = 0; i < _size; ++i) {
      result[i] = _data[i] - rhs;
    }
    return result;
  }

  // Subtraction with int
  ComplexVector operator-(int rhs) const {
    ComplexVector result(_size);
    for (int i = 0; i < _size; ++i) {
      result[i] = _data[i] - rhs;
    }
    return result;
  }

  // Subtraction with float
  ComplexVector operator-(float rhs) const {
    ComplexVector result(_size);
    for (int i = 0; i < _size; ++i) {
      result[i] = _data[i] - rhs;
    }
    return result;
  }

  // Multiplication with double
  ComplexVector operator*(double scale) const {
    ComplexVector result(_size);
    for (int i = 0; i < _size; ++i) {
      result[i] = _data[i] * scale;
    }
    return result;
  }

  // Multiplication with int
  ComplexVector operator*(int scale) const {
    ComplexVector result(_size);
    for (int i = 0; i < _size; ++i) {
      result[i] = _data[i] * scale;
    }
    return result;
  }

  // Multiplication with float
  ComplexVector operator*(float scale) const {
    ComplexVector result(_size);
    for (int i = 0; i < _size; ++i) {
      result[i] = _data[i] * scale;
    }
    return result;
  }

  // Arithmetic operators with Complex

  // Addition with Complex
  ComplexVector operator+(const ComplexVector &rhs) const {
    ComplexVector result(_size);
    for (int i = 0; i < _size; ++i) {
      result[i] = _data[i] + rhs[i];
    }
    return result;
  }

  // Subtraction with Complex
  ComplexVector operator-(const ComplexVector &rhs) const {
    ComplexVector result(_size);
    for (int i = 0; i < _size; ++i) {
      result[i] = _data[i] - rhs[i];
    }
    return result;
  }

  // Multiplication with Complex
  ComplexVector operator*(const ComplexVector &rhs) const {
    ComplexVector result(_size);
    for (int i = 0; i < _size; ++i) {
      result[i] = _data[i] * rhs[i];
    }
    return result;
  }

  // Scaling operators

  // Scaling with double
  ComplexVector operator*(double scale) {
    for (int i = 0; i < _size; ++i) {
      _data[i] *= scale;
    }
    return *this;
  }

  // Scaling with int
  ComplexVector operator*(int scale) {
    for (int i = 0; i < _size; ++i) {
      _data[i] *= scale;
    }
    return *this;
  }

  // Scaling with float
  ComplexVector operator*(float scale) {
    for (int i = 0; i < _size; ++i) {
      _data[i] *= scale;
    }
    return *this;
  }

  // Arithmetic operators with assignment

  // Addition with assignment
  ComplexVector &operator+=(const ComplexVector &rhs) {
    for (int i = 0; i < _size; ++i) {
      _data[i] += rhs[i];
    }
    return *this;
  }

  // Subtraction with assignment
  ComplexVector &operator-=(const ComplexVector &rhs) {
    for (int i = 0; i < _size; ++i) {
      _data[i] -= rhs[i];
    }
    return *this;
  }

  // Multiplication with assignment
  ComplexVector &operator*=(const ComplexVector &rhs) {
    for (int i = 0; i < _size; ++i) {
      _data[i] *= rhs[i];
    }
    return *this;
  }

  // Scaling operators with assignment

  // Scaling with double with assignment
  ComplexVector &operator*=(double scale) {
    for (int i = 0; i < _size; ++i) {
      _data[i] *= scale;
    }
    return *this;
  }

  // Scaling with int with assignment
  ComplexVector &operator*=(int scale) {
    for (int i = 0; i < _size; ++i) {
      _data[i] *= scale;
    }
    return *this;
  }

  // Scaling with float with assignment
  ComplexVector &operator*=(float scale) {
    for (int i = 0; i < _size; ++i) {
      _data[i] *= scale;
    }
    return *this;
  }

  // Accessors
  Complex &operator[](int index) { return _data[index]; }

  const Complex &operator[](int index) const { return _data[index]; }

  // String representation
  std::string to_string() const {
    std::string result;
    for (int i = 0; i < _size; ++i) {
      result += _data[i].to_string() + " ";
    }
    return result;
  }

private:
  Complex *_data;
  int _size;
};

#endif