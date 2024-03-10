// #ifndef _COMPLEX_VECTOR_H
// #define _COMPLEX_VECTOR_H
// #pragma optimize(5)
// #include "./qcu.h"

// // Complex vector class
// class ComplexVector {
// public:
//   // Data members
//   Complex *_data;
//   int _size;

//   // Constructors

//   // Default constructor
//   __device__ ComplexVector() : _data(nullptr), _size(0) {}

//   // Constructor with size
//   __device__ ComplexVector(int size) : _size(size) {
//     cudaMallocManaged(&_data, _size * sizeof(Complex));
//   }

//   // Constructor with pointer to data and size
//   __device__ ComplexVector(void *data, int size)
//       : _data((static_cast<Complex *>(data))), _size(size) {}

//   // Copy constructor
//   __device__ ComplexVector(const ComplexVector &other)
//       : _data(new Complex[other._size]), _size(other._size) {
//     for (int i = 0; i < _size; ++i) {
//       _data[i] = other._data[i];
//     }
//   }

//   // Move constructor
//   __device__ ComplexVector(ComplexVector &&other) noexcept
//       : _data(other._data), _size(other._size) {}

//   // Destructor
//   ~ComplexVector() {
//     if (_data != nullptr) {
//       _data = nullptr;
//       // delete[] _data;
//       checkCudaErrors(cudaFree(_data));
//     }
//   }
//   // __device__ ~ComplexVector() {}

//   // Element access
//   __device__ Complex &operator[](int index) { return _data[index]; }

//   __device__ const Complex &operator[](int index) const { return _data[index]; }

//   // Arithmetic operators

//   // Uses the AVX vectorization instruction set if available
//   __device__ ComplexVector operator+(const ComplexVector &rhs) const {
// #if defined(__AVX__)
//     __m256d this_vector = _mm256_load_pd(_data);
//     __m256d other_vector = _mm256_load_pd(rhs._data);
//     __m256d result_vector = _mm256_add_pd(this_vector, other_vector);
//     ComplexVector result(_size);
//     _mm256_store_pd(result._data, result_vector);
//     return result;
// #else
//     ComplexVector result(_size);
//     for (int i = 0; i < _size; ++i) {
//       result[i] = _data[i] + rhs[i];
//     }
//     return result;
// #endif
//   }
//   __device__ ComplexVector operator+(double rhs) const {
//     ComplexVector result(_size);
//     for (int i = 0; i < _size; ++i) {
//       result[i] = _data[i] + rhs;
//     }
//     return result;
//   }
//   // Uses the AVX vectorization instruction set if available
//   __device__ ComplexVector operator-(const ComplexVector &rhs) const {
// #if defined(__AVX__)
//     __m256d this_vector = _mm256_load_pd(_data);
//     __m256d other_vector = _mm256_load_pd(rhs._data);
//     __m256d result_vector = _mm256_sub_pd(this_vector, other_vector);
//     ComplexVector result(_size);
//     _mm256_store_pd(result._data, result_vector);
//     return result;
// #else
//     ComplexVector result(_size);
//     for (int i = 0; i < _size; ++i) {
//       result[i] = _data[i] - rhs[i];
//     }
//     return result;
// #endif
//   }
//   __device__ ComplexVector operator-(double rhs) const {
//     ComplexVector result(_size);
//     for (int i = 0; i < _size; ++i) {
//       result[i] = _data[i] - rhs;
//     }
//     return result;
//   }
//   __device__ ComplexVector operator*(const ComplexVector &rhs) const {
//     ComplexVector result(_size);
//     for (int i = 0; i < _size; ++i) {
//       result[i] = _data[i] * rhs[i];
//     }
//     return result;
//   }
//   // Uses the AVX vectorization instruction set if available
//   __device__ ComplexVector operator*(double rhs) const {
// #if defined(__AVX__)
//     __m256d this_vector = _mm256_load_pd(_data);
//     __m256d scaled_vector = _mm256_mul_pd(this_vector, _mm256_set1_pd(rhs));
//     ComplexVector result(_size);
//     _mm256_store_pd(result._data, scaled_vector);
//     return result;
// #else
//     ComplexVector result(_size);
//     for (int i = 0; i < _size; ++i) {
//       result[i] = _data[i] * rhs;
//     }
//     return result;
// #endif
//   }
//   __device__ ComplexVector operator/(const ComplexVector &rhs) const {
//     ComplexVector result(_size);
//     for (int i = 0; i < _size; ++i) {
//       result[i] = _data[i] / rhs[i];
//     }
//     return result;
//   }
//   __device__ ComplexVector operator/(double rhs) const {
//     ComplexVector result(_size);
//     for (int i = 0; i < _size; ++i) {
//       result[i] = _data[i] / rhs;
//     }
//     return result;
//   }

//   // Arithmetic operators with assignment
//   __device__ ComplexVector &operator=(const ComplexVector &rhs) {
//     for (int i = 0; i < _size; ++i) {
//       _data[i] = rhs[i];
//     }
//     return *this;
//   }
//   __device__ ComplexVector &operator=(const double &rhs) {
//     for (int i = 0; i < _size; ++i) {
//       _data[i] = rhs;
//     }
//     return *this;
//   }
//   __device__ ComplexVector &operator+=(const ComplexVector &rhs) {
//     for (int i = 0; i < _size; ++i) {
//       _data[i] += rhs[i];
//     }
//     return *this;
//   }
//   __device__ ComplexVector &operator+=(const double &rhs) {
//     for (int i = 0; i < _size; ++i) {
//       _data[i] += rhs;
//     }
//     return *this;
//   }
//   __device__ ComplexVector &operator-=(const ComplexVector &rhs) {
//     for (int i = 0; i < _size; ++i) {
//       _data[i] -= rhs[i];
//     }
//     return *this;
//   }
//   __device__ ComplexVector &operator-=(const double &rhs) {
//     for (int i = 0; i < _size; ++i) {
//       _data[i] -= rhs;
//     }
//     return *this;
//   }
//   __device__ ComplexVector &operator*=(const ComplexVector &rhs) {
//     for (int i = 0; i < _size; ++i) {
//       _data[i] *= rhs[i];
//     }
//     return *this;
//   }
//   __device__ ComplexVector &operator*=(const double &rhs) {
//     for (int i = 0; i < _size; ++i) {
//       _data[i] *= rhs;
//     }
//     return *this;
//   }
//   __device__ ComplexVector &operator/=(const ComplexVector &rhs) {
//     for (int i = 0; i < _size; ++i) {
//       _data[i] /= rhs[i];
//     }
//     return *this;
//   }
//   __device__ ComplexVector &operator/=(const double &rhs) {
//     for (int i = 0; i < _size; ++i) {
//       _data[i] /= rhs;
//     }
//     return *this;
//   }

//   // Unary minus
//   __device__ ComplexVector operator-() const {
//     return (*this) * (-1.0);
//   }

//   // Conjugate
//   __device__ ComplexVector conj() const {
//     ComplexVector result(_size);
//     for (int i = 0; i < _size; ++i) {
//       result[i] = _data[i].conj();
//     }
//     return result;
//   }

//   // Multiplication with i
//   __device__ ComplexVector mul_i() const {
//     ComplexVector result(_size);
//     for (int i = 0; i < _size; ++i) {
//       result[i] = _data[i].mul_i();
//     }
//     return result;
//   }

//   // Multiplication with ii
//   __device__ ComplexVector mul_ii() const {
//     ComplexVector result(_size);
//     for (int i = 0; i < _size; ++i) {
//       result[i] = _data[i].mul_ii();
//     }
//     return result;
//   }

//   // Multiplication with iii
//   __device__ ComplexVector mul_iii() const {
//     ComplexVector result(_size);
//     for (int i = 0; i < _size; ++i) {
//       result[i] = _data[i].mul_iii();
//     }
//     return result;
//   }

//   // Randomly initializes the elements of the vector
//   __host__ void init_random(unsigned seed = 0) {
//     srand(seed);
//     for (int i = 0; i < _size; i++) {
//       _data[i] = Complex(std::rand() / 1e9, std::rand() / 1e9);
//     }
//   }

//   // String representation
//   __host__ std::string to_string() const {
//     std::string result;
//     for (int i = 0; i < OUTPUT_SIZE; ++i) {
//       result += _data[i].to_string() + " ";
//     }
//     result += " ...";
//     return result;
//   }
// };

// #endif