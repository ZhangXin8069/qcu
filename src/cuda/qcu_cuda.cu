#include <filesystem>
#include <limits>
#include <sys/types.h>
#pragma nv_verbose
#pragma optimize(5)
#include "qcu.h"
#include <chrono>
#include <cmath>
#include <cstdio>

#define BLOCK_SIZE 256
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

#define checkCudaErrors(err)                                                   \
  {                                                                            \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr,                                                          \
              "checkCudaErrors() API error = %04d \"%s\" from file <%s>, "     \
              "line %i.\n",                                                    \
              err, cudaGetErrorString(err), __FILE__, __LINE__);               \
      exit(-1);                                                                \
    }                                                                          \
  }

#define give_value(U, zero, n)                                                 \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] = zero;                                                             \
    }                                                                          \
  }

#define give_ptr(U, origin_U, n)                                               \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] = origin_U[i];                                                      \
    }                                                                          \
  }

#define give_rand(input_matrix, size)                                          \
  {                                                                            \
    for (int i = 0; i < size; i++) {                                           \
      input_matrix[i].real = static_cast<double>(rand()) / RAND_MAX;           \
      input_matrix[i].imag = static_cast<double>(rand()) / RAND_MAX;           \
    }                                                                          \
  }

#define give_u(tmp, tmp_U)                                                     \
  {                                                                            \
    for (int i = 0; i < 6; i++) {                                              \
      tmp[i] = tmp_U[i];                                                       \
    }                                                                          \
    tmp[6] = (tmp[1] * tmp[5] - tmp[2] * tmp[4]).conj();                       \
    tmp[7] = (tmp[2] * tmp[3] - tmp[0] * tmp[5]).conj();                       \
    tmp[8] = (tmp[0] * tmp[4] - tmp[1] * tmp[3]).conj();                       \
  }

#define add(U, tmp, n)                                                         \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] += tmp[i];                                                          \
    }                                                                          \
  }

#define subt(U, tmp, n)                                                        \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] -= tmp[i];                                                          \
    }                                                                          \
  }

#define mult(tmp0, tmp1, tmp2, tmp3, zero)                                     \
  {                                                                            \
    for (int c0 = 0; c0 < 3; c0++) {                                           \
      for (int c1 = 0; c1 < 3; c1++) {                                         \
        tmp0 = zero;                                                           \
        for (int cc = 0; cc < 3; cc++) {                                       \
          tmp0 += tmp1[c0 * 3 + cc] * tmp2[cc * 3 + c1];                       \
        }                                                                      \
        tmp3[c0 * 3 + c1] = tmp0;                                              \
      }                                                                        \
    }                                                                          \
  }

#define mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero)                         \
  {                                                                            \
    for (int c0 = 0; c0 < 3; c0++) {                                           \
      for (int c1 = 0; c1 < 3; c1++) {                                         \
        tmp0 = zero;                                                           \
        for (int cc = 0; cc < 3; cc++) {                                       \
          tmp0 += tmp1[c0 * 3 + cc] * tmp2[cc * 3 + c1];                       \
        }                                                                      \
        tmp3[c0 * 3 + c1] = tmp0;                                              \
      }                                                                        \
    }                                                                          \
  }

#define mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero)                          \
  {                                                                            \
    for (int c0 = 0; c0 < 3; c0++) {                                           \
      for (int c1 = 0; c1 < 3; c1++) {                                         \
        tmp0 = zero;                                                           \
        for (int cc = 0; cc < 3; cc++) {                                       \
          tmp0 += tmp1[c0 * 3 + cc] * tmp2[c1 * 3 + cc].conj();                \
        }                                                                      \
        tmp3[c0 * 3 + c1] = tmp0;                                              \
      }                                                                        \
    }                                                                          \
  }

#define mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero)                          \
  {                                                                            \
    for (int c0 = 0; c0 < 3; c0++) {                                           \
      for (int c1 = 0; c1 < 3; c1++) {                                         \
        tmp0 = zero;                                                           \
        for (int cc = 0; cc < 3; cc++) {                                       \
          tmp0 += tmp1[cc * 3 + c0].conj() * tmp2[cc * 3 + c1];                \
        }                                                                      \
        tmp3[c0 * 3 + c1] = tmp0;                                              \
      }                                                                        \
    }                                                                          \
  }

#define mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero)                           \
  {                                                                            \
    for (int c0 = 0; c0 < 3; c0++) {                                           \
      for (int c1 = 0; c1 < 3; c1++) {                                         \
        tmp0 = zero;                                                           \
        for (int cc = 0; cc < 3; cc++) {                                       \
          tmp0 += tmp1[cc * 3 + c0].conj() * tmp2[c1 * 3 + cc].conj();         \
        }                                                                      \
        tmp3[c0 * 3 + c1] = tmp0;                                              \
      }                                                                        \
    }                                                                          \
  }

#define inverse(input_matrix, inverse_matrix, augmented_matrix, pivot, factor, \
                size)                                                          \
  {                                                                            \
    for (int i = 0; i < size; i++) {                                           \
      for (int j = 0; j < size; j++) {                                         \
        inverse_matrix[i * size + j] = input_matrix[i * size + j];             \
        augmented_matrix[i * 2 * size + j] = inverse_matrix[i * size + j];     \
      }                                                                        \
      augmented_matrix[i * 2 * size + size + i] = 1.0;                         \
    }                                                                          \
    for (int i = 0; i < size; i++) {                                           \
      pivot = augmented_matrix[i * 2 * size + i];                              \
      for (int j = 0; j < 2 * size; j++) {                                     \
        augmented_matrix[i * 2 * size + j] /= pivot;                           \
      }                                                                        \
      for (int j = 0; j < size; j++) {                                         \
        if (j != i) {                                                          \
          factor = augmented_matrix[j * 2 * size + i];                         \
          for (int k = 0; k < 2 * size; ++k) {                                 \
            augmented_matrix[j * 2 * size + k] -=                              \
                factor * augmented_matrix[i * 2 * size + k];                   \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    for (int i = 0; i < size; i++) {                                           \
      for (int j = 0; j < size; j++) {                                         \
        inverse_matrix[i * size + j] =                                         \
            augmented_matrix[i * 2 * size + size + j];                         \
      }                                                                        \
    }                                                                          \
  }

#define inverse_clover(input_matrix, inverse_matrix, augmented_matrix, pivot,  \
                       factor)                                                 \
  {                                                                            \
    for (int s0 = 0; s0 < 4; s0++) {                                           \
      for (int c0 = 0; c0 < 3; c0++) {                                         \
        for (int s1 = 0; s1 < 4; s1++) {                                       \
          for (int c1 = 0; c1 < 3; c1++) {                                     \
            inverse_matrix[s0 * 36 + s1 * 9 + c0 * 3 + c1] =                   \
                input_matrix[s0 * 36 + s1 * 9 + c0 * 3 + c1];                  \
            augmented_matrix[s0 * 72 + s1 * 9 + c0 * 3 + c1] =                 \
                inverse_matrix[s0 * 36 + s1 * 9 + c0 * 3 + c1];                \
          }                                                                    \
        }                                                                      \
        augmented_matrix[s0 * 72 + (4 + s0) * 9 + c0 * 3 + c0] = 1.0;          \
      }                                                                        \
    }                                                                          \
    for (int s0 = 0; s0 < 4; s0++) {                                           \
      for (int c0 = 0; c0 < 3; c0++) {                                         \
        pivot = augmented_matrix[s0 * 72 + s0 * 9 + c0 * 3 + c0];              \
        for (int s1 = 0; s1 < 8; s1++) {                                       \
          for (int c1 = 0; c1 < 3; c1++) {                                     \
            augmented_matrix[s0 * 72 + s1 * 9 + c0 * 3 + c1] /= pivot;         \
          }                                                                    \
        }                                                                      \
        for (int s1 = 0; s1 < 4; s1++) {                                       \
          for (int c1 = 0; c1 < 3; c1++) {                                     \
            if ((s0 != s1) || (c0 != c1)) {                                    \
              factor = augmented_matrix[s1 * 72 + s0 * 9 + c1 * 3 + c0];       \
              for (int ss1 = 0; ss1 < 8; ss1++) {                              \
                for (int cc1 = 0; cc1 < 3; cc1++) {                            \
                  augmented_matrix[s1 * 72 + ss1 * 9 + c1 * 3 + cc1] -=        \
                      factor *                                                 \
                      augmented_matrix[s0 * 72 + ss1 * 9 + c0 * 3 + cc1];      \
                }                                                              \
              }                                                                \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    for (int s0 = 0; s0 < 4; s0++) {                                           \
      for (int c0 = 0; c0 < 3; c0++) {                                         \
        for (int s1 = 0; s1 < 4; s1++) {                                       \
          for (int c1 = 0; c1 < 3; c1++) {                                     \
            inverse_matrix[s0 * 36 + s1 * 9 + c0 * 3 + c1] =                   \
                augmented_matrix[s0 * 72 + (4 + s1) * 9 + c0 * 3 + c1];        \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

#define print_gauge(input_gauge)                                               \
  {                                                                            \
    printf("############\n");                                                  \
    printf("[");                                                               \
    printf("[%.9lf,%.9lf]", input_gauge[0].real, input_gauge[0].imag);         \
    printf("[%.9lf,%.9lf]", input_gauge[1].real, input_gauge[1].imag);         \
    printf("[%.9lf,%.9lf]", input_gauge[2].real, input_gauge[2].imag);         \
    printf("]\n");                                                             \
    printf("[");                                                               \
    printf("[%.9lf,%.9lf]", input_gauge[3].real, input_gauge[3].imag);         \
    printf("[%.9lf,%.9lf]", input_gauge[4].real, input_gauge[4].imag);         \
    printf("[%.9lf,%.9lf]", input_gauge[5].real, input_gauge[5].imag);         \
    printf("]\n");                                                             \
    printf("[");                                                               \
    printf("[%.9lf,%.9lf]", input_gauge[6].real, input_gauge[6].imag);         \
    printf("[%.9lf,%.9lf]", input_gauge[7].real, input_gauge[7].imag);         \
    printf("[%.9lf,%.9lf]", input_gauge[8].real, input_gauge[8].imag);         \
    printf("]\n");                                                             \
  }

#define print_fermi(input_fermi)                                               \
  {                                                                            \
    int tmp;                                                                   \
    for (int s = 0; s < 4; s++) {                                              \
      printf("######S%.1d######\n", s);                                        \
      tmp = s * 9;                                                             \
      printf("[");                                                             \
      printf("[%.9lf,%.9lf]", input_fermi[tmp].real, input_fermi[tmp].imag);   \
      printf("[%.9lf,%.9lf]", input_fermi[tmp + 1].real,                       \
             input_fermi[tmp + 1].imag);                                       \
      printf("[%.9lf,%.9lf]", input_fermi[tmp + 2].real,                       \
             input_fermi[tmp + 2].imag);                                       \
      printf("]\n");                                                           \
      printf("[");                                                             \
      printf("[%.9lf,%.9lf]", input_fermi[tmp + 3].real,                       \
             input_fermi[tmp + 3].imag);                                       \
      printf("[%.9lf,%.9lf]", input_fermi[tmp + 4].real,                       \
             input_fermi[tmp + 4].imag);                                       \
      printf("[%.9lf,%.9lf]", input_fermi[tmp + 5].real,                       \
             input_fermi[tmp + 5].imag);                                       \
      printf("]\n");                                                           \
      printf("[");                                                             \
      printf("[%.9lf,%.9lf]", input_fermi[tmp + 6].real,                       \
             input_fermi[tmp + 6].imag);                                       \
      printf("[%.9lf,%.9lf]", input_fermi[tmp + 7].real,                       \
             input_fermi[tmp + 7].imag);                                       \
      printf("[%.9lf,%.9lf]", input_fermi[tmp + 8].real,                       \
             input_fermi[tmp + 8].imag);                                       \
      printf("]\n");                                                           \
    }                                                                          \
  }

#define print_clover(input_clover)                                             \
  {                                                                            \
    int tmp;                                                                   \
    for (int s0 = 0; s0 < 4; s0++) {                                           \
      for (int s1 = 0; s1 < 4; s1++) {                                         \
        printf("######S%.1d,%.1dS######\n", s0, s1);                           \
        tmp = s0 * 36 + s1 * 9;                                                \
        printf("[");                                                           \
        printf("[%.9lf,%.9lf]", input_clover[tmp].real,                        \
               input_clover[tmp].imag);                                        \
        printf("[%.9lf,%.9lf]", input_clover[tmp + 1].real,                    \
               input_clover[tmp + 1].imag);                                    \
        printf("[%.9lf,%.9lf]", input_clover[tmp + 2].real,                    \
               input_clover[tmp + 2].imag);                                    \
        printf("]\n");                                                         \
        printf("[");                                                           \
        printf("[%.9lf,%.9lf]", input_clover[tmp + 3].real,                    \
               input_clover[tmp + 3].imag);                                    \
        printf("[%.9lf,%.9lf]", input_clover[tmp + 4].real,                    \
               input_clover[tmp + 4].imag);                                    \
        printf("[%.9lf,%.9lf]", input_clover[tmp + 5].real,                    \
               input_clover[tmp + 5].imag);                                    \
        printf("]\n");                                                         \
        printf("[");                                                           \
        printf("[%.9lf,%.9lf]", input_clover[tmp + 6].real,                    \
               input_clover[tmp + 6].imag);                                    \
        printf("[%.9lf,%.9lf]", input_clover[tmp + 7].real,                    \
               input_clover[tmp + 7].imag);                                    \
        printf("[%.9lf,%.9lf]", input_clover[tmp + 8].real,                    \
               input_clover[tmp + 8].imag);                                    \
        printf("]\n");                                                         \
      }                                                                        \
    }                                                                          \
  }

__global__ void wilson_dslash(void *device_U, void *device_src,
                              void *device_dest, int device_lat_x,
                              const int device_lat_y, const int device_lat_z,
                              const int device_lat_t, const int device_parity) {
  register int parity = blockIdx.x * blockDim.x + threadIdx.x;
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  const int lat_t = device_lat_t;
  const int lat_xcc = lat_x * 9;
  const int lat_yxcc = lat_y * lat_xcc;
  const int lat_zyxcc = lat_z * lat_yxcc;
  const int lat_tzyxcc = lat_t * lat_zyxcc;
  const int lat_xsc = lat_x * 12;
  const int lat_yxsc = lat_y * lat_xsc;
  const int lat_zyxsc = lat_z * lat_yxsc;
  register int move;
  move = lat_x * lat_y * lat_z;
  const int t = parity / move;
  parity -= t * move;
  move = lat_x * lat_y;
  const int z = parity / move;
  parity -= z * move;
  const int y = parity / lat_x;
  const int x = parity - y * lat_x;
  parity = device_parity;
  const int oe = (y + z + t) % 2;
  register LatticeComplex I(0.0, 1.0);
  register LatticeComplex zero(0.0, 0.0);
  register LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) + t * lat_zyxcc +
       z * lat_yxcc + y * lat_xcc + x * 9);
  register LatticeComplex *origin_src =
      ((static_cast<LatticeComplex *>(device_src)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  register LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  register LatticeComplex *tmp_U;
  register LatticeComplex *tmp_src;
  register LatticeComplex tmp0(0.0, 0.0);
  register LatticeComplex tmp1(0.0, 0.0);
  register LatticeComplex U[9];
  register LatticeComplex src[12];
  register LatticeComplex dest[12];
  // just wilson(Sum part)
  give_value(dest, zero, 12);
  {
    // x-1
    move = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move * 9 + (1 - parity) * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * 12);
    give_ptr(src, tmp_src, 12);
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] + src[c1 + 9] * I) * U[c1 * 3 + c0].conj();
        tmp1 += (src[c1 + 3] + src[c1 + 6] * I) * U[c1 * 3 + c0].conj();
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] -= tmp1 * I;
      dest[c0 + 9] -= tmp0 * I;
    }
  }
  {
    // x+1
    move = (1 - (x == lat_x - 1) * lat_x) * (oe != parity);
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * 12);
    give_ptr(src, tmp_src, 12);
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] - src[c1 + 9] * I) * U[c0 * 3 + c1];
        tmp1 += (src[c1 + 3] - src[c1 + 6] * I) * U[c0 * 3 + c1];
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] += tmp1 * I;
      dest[c0 + 9] += tmp0 * I;
    }
  }
  {
    // y-1
    move = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_xsc);
    give_ptr(src, tmp_src, 12);
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] - src[c1 + 9]) * U[c1 * 3 + c0].conj();
        tmp1 += (src[c1 + 3] + src[c1 + 6]) * U[c1 * 3 + c0].conj();
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] += tmp1;
      dest[c0 + 9] -= tmp0;
    }
  }
  {
    // y+1
    move = 1 - (y == lat_y - 1) * lat_y;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_xsc);
    give_ptr(src, tmp_src, 12);
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] + src[c1 + 9]) * U[c0 * 3 + c1];
        tmp1 += (src[c1 + 3] - src[c1 + 6]) * U[c0 * 3 + c1];
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] -= tmp1;
      dest[c0 + 9] += tmp0;
    }
  }
  {
    // z-1
    move = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_yxsc);
    give_ptr(src, tmp_src, 12);
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] + src[c1 + 6] * I) * U[c1 * 3 + c0].conj();
        tmp1 += (src[c1 + 3] - src[c1 + 9] * I) * U[c1 * 3 + c0].conj();
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] -= tmp0 * I;
      dest[c0 + 9] += tmp1 * I;
    }
  }
  {
    // z+1
    move = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_yxsc);
    give_ptr(src, tmp_src, 12);
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] - src[c1 + 6] * I) * U[c0 * 3 + c1];
        tmp1 += (src[c1 + 3] + src[c1 + 9] * I) * U[c0 * 3 + c1];
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] += tmp0 * I;
      dest[c0 + 9] -= tmp1 * I;
    }
  }
  {
    // t-1
    move = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_zyxsc);
    give_ptr(src, tmp_src, 12);
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] + src[c1 + 6]) * U[c1 * 3 + c0].conj();
        tmp1 += (src[c1 + 3] + src[c1 + 9]) * U[c1 * 3 + c0].conj();
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] += tmp0;
      dest[c0 + 9] += tmp1;
    }
  }
  {
    // t+1
    move = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_zyxsc);
    give_ptr(src, tmp_src, 12);
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] - src[c1 + 6]) * U[c0 * 3 + c1];
        tmp1 += (src[c1 + 3] - src[c1 + 9]) * U[c0 * 3 + c1];
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] -= tmp0;
      dest[c0 + 9] -= tmp1;
    }
  }
  give_ptr(origin_dest, dest, 12);
}

__global__ void make_clover(void *device_U, void *device_clover,
                            int device_lat_x, const int device_lat_y,
                            const int device_lat_z, const int device_lat_t,
                            const int device_parity) {
  register int parity = blockIdx.x * blockDim.x + threadIdx.x;
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  const int lat_t = device_lat_t;
  const int lat_xcc = lat_x * 9;
  const int lat_yxcc = lat_y * lat_xcc;
  const int lat_zyxcc = lat_z * lat_yxcc;
  const int lat_tzyxcc = lat_t * lat_zyxcc;
  register int move0;
  register int move1;
  move0 = lat_x * lat_y * lat_z;
  const int t = parity / move0;
  parity -= t * move0;
  move0 = lat_x * lat_y;
  const int z = parity / move0;
  parity -= z * move0;
  const int y = parity / lat_x;
  const int x = parity - y * lat_x;
  const int oe = (y + z + t) % 2;
  register LatticeComplex I(0.0, 1.0);
  register LatticeComplex zero(0.0, 0.0);
  register LatticeComplex tmp0(0.0, 0.0);
  register LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) + t * lat_zyxcc +
       z * lat_yxcc + y * lat_xcc + x * 9);
  register LatticeComplex *origin_clover =
      ((static_cast<LatticeComplex *>(device_clover)) + t * lat_zyxcc * 16 +
       z * lat_yxcc * 16 + y * lat_xcc * 16 + x * 144);
  register LatticeComplex *tmp_U;
  register LatticeComplex tmp1[9];
  register LatticeComplex tmp2[9];
  register LatticeComplex tmp3[9];
  register LatticeComplex U[9];
  register LatticeComplex clover[144];
  // sigmaF
  {
    parity = device_parity;
    give_value(clover, zero, 144);
    give_value(origin_clover, zero, 144);
    give_value(tmp1, zero, 9);
    give_value(tmp2, zero, 9);
  }
  // XY
  give_value(U, zero, 9);
  {
    //// x,y,z,t;x
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x+1,y,z,t;y
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe != parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 2 + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z,t;x;dag
    move0 = 1 - (y == lat_y - 1) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t;y
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y+1,z,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = 1 - (y == lat_y - 1) * lat_y;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;y;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 2 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x-1,y,z,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y-1,z,t;y;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y-1,z,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y-1,z,t;y;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z,t;x
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y-1,z,t;y
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe == parity);
    move1 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[45 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
        clover[90 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[135 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
      }
    }
  }
  // XZ
  give_value(U, zero, 9);
  {
    //// x,y,z,t;x
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x+1,y,z,t;z
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe != parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 4 + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t;x;dag
    move0 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t;z
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z+1,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;z;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 4 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x-1,y,z,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z-1,t;z;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z-1,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z-1,t;z;dag
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z-1,t;x
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y,z-1,t;z
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe == parity);
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[9 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-1);
        clover[36 + c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj());
        clover[99 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-1);
        clover[126 + c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj());
      }
    }
  }
  // XT
  give_value(U, zero, 9);
  {
    //// x,y,z,t;x
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x+1,y,z,t;t
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe != parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 6 + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;x;dag
    move0 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t;t
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z,t+1;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;t;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 6 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x-1,y,z,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z,t-1;t;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t-1;x
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t-1;t;dag
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z,t-1;x
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y,z,t-1;t
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe == parity);
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[9 + c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
        clover[36 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
        clover[99 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[126 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
      }
    }
  }
  // YZ
  give_value(U, zero, 9);
  {
    //// x,y,z,t;y
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y+1,z,t;z
    move0 = 1 - (y == lat_y - 1) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t;y;dag
    move0 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t;z
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z+1,t;y;dag
    move0 = -1 + (y == 0) * lat_y;
    move1 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t;z;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y-1,z,t;y;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z-1,t;z;dag
    move0 = -1 + (y == 0) * lat_y;
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z-1,t;y
    move0 = -1 + (y == 0) * lat_y;
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z-1,t;z;dag
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z-1,t;y
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z-1,t;z
    move0 = 1 - (y == lat_y - 1) * lat_y;
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[9 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[36 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[99 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[126 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
      }
    }
  }
  // YT
  give_value(U, zero, 9);
  {
    //// x,y,z,t;y
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y+1,z,t;t
    move0 = 1 - (y == lat_y - 1) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;y;dag
    move0 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t;t
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z,t+1;y;dag
    move0 = -1 + (y == 0) * lat_y;
    move1 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t;t;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y-1,z,t;y;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z,t-1;t;dag
    move0 = -1 + (y == 0) * lat_y;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t-1;y
    move0 = -1 + (y == 0) * lat_y;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t-1;t;dag
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z,t-1;y
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z,t-1;t
    move0 = 1 - (y == lat_y - 1) * lat_y;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[9 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-1);
        clover[36 + c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj());
        clover[99 + c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj());
        clover[126 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-1);
      }
    }
  }

  // ZT
  give_value(U, zero, 9);
  {
    //// x,y,z,t;z
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z+1,t;t
    move0 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;z;dag
    move0 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t;t
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z-1,t+1;z;dag
    move0 = -1 + (z == 0) * lat_z;
    move1 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z-1,t;t;dag
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z-1,t;z;dag
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z-1,t-1;t;dag
    move0 = -1 + (z == 0) * lat_z;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z-1,t-1;z
    move0 = -1 + (z == 0) * lat_z;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t-1;t;dag
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z,t-1;z
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t-1;t
    move0 = 1 - (z == lat_z - 1) * lat_z;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
        clover[45 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[90 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[135 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
      }
    }
  }
  {
    // A=1+T
    LatticeComplex one(1.0, 0);
    for (int i = 0; i < 144; i++) {
      clover[i] *= -0.125;
    }
    for (int s = 0; s < 4; s++) {
      for (int c = 0; c < 3; c++) {
        clover[s * 45 + c * 4] += one;
      }
    }
    for (int i = 0; i < 144; i++) {
      origin_clover[i] = clover[i];
    }
  }
}

__global__ void give_clover(void *device_clover, void *device_dest,
                            int device_lat_x, const int device_lat_y,
                            const int device_lat_z) {
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  register LatticeComplex zero(0.0, 0.0);
  register LatticeComplex I(0.0, 1.0);
  register LatticeComplex tmp0;
  register int tmp1;
  register int tmp2 = blockIdx.x * blockDim.x + threadIdx.x;
  tmp1 = lat_x * lat_y * lat_z;
  const int t = tmp2 / tmp1;
  tmp2 -= t * tmp1;
  tmp1 = lat_x * lat_y;
  const int z = tmp2 / tmp1;
  tmp2 -= z * tmp1;
  const int y = tmp2 / lat_x;
  const int x = tmp2 - y * lat_x;
  LatticeComplex pivot;
  LatticeComplex factor;
  LatticeComplex *origin_clover =
      ((static_cast<LatticeComplex *>(device_clover)) +
       t * lat_z * lat_y * lat_x * 144 + z * lat_y * lat_x * 144 +
       y * lat_x * 144 + x * 144);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) +
       t * lat_z * lat_y * lat_x * 12 + z * lat_y * lat_x * 12 +
       y * lat_x * 12 + x * 12);
  LatticeComplex clover[144];
  LatticeComplex augmented_clover[288];
  LatticeComplex dest[12];
  LatticeComplex tmp_dest[12];
  give_ptr(dest, origin_dest, 12);
  give_ptr(clover, origin_clover, 144);
  inverse_clover(clover, clover, augmented_clover, pivot, factor);
  {
    for (int s0 = 0; s0 < 4; s0++) {
      for (int c0 = 0; c0 < 3; c0++) {
        tmp0 = zero;
        for (int s1 = 0; s1 < 4; s1++) {
          for (int c1 = 0; c1 < 3; c1++) {
            tmp0 += clover[s0 * 36 + s1 * 9 + c0 * 3 + c1] * dest[s1 * 3 + c1];
          }
        }
        tmp_dest[s0 * 3 + c0] = tmp0;
      }
    }
    give_ptr(origin_dest, tmp_dest, 12);
  }
}

void dslashQcu(void *fermion_out, void *fermion_in, void *gauge,
               QcuParam *param, int parity) {
  int lat_x = param->lattice_size[0] >> 1;
  int lat_y = param->lattice_size[1];
  int lat_z = param->lattice_size[2];
  int lat_t = param->lattice_size[3];
  void *clover;
  checkCudaErrors(cudaMalloc(&clover, (lat_t * lat_z * lat_y * lat_x * 144) *
                                          sizeof(LatticeComplex)));
  cudaError_t err;
  dim3 gridDim(lat_x * lat_y * lat_z * lat_t / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  {
    // wilson dslash
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    wilson_dslash<<<gridDim, blockDim>>>(gauge, fermion_in, fermion_out, lat_x,
                                         lat_y, lat_z, lat_t, parity);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf(
        "wilson dslash total time: (without malloc free memcpy) : %.9lf sec\n",
        double(duration) / 1e9);
  }
  {
    // just clover
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    make_clover<<<gridDim, blockDim>>>(gauge, clover, lat_x, lat_y, lat_z,
                                       lat_t, parity);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf("just clover total time: (without malloc free memcpy) :%.9lf sec\n ",
           double(duration) / 1e9);
  }
  {
    // give clover
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    give_clover<<<gridDim, blockDim>>>(clover, fermion_out, lat_x, lat_y,
                                       lat_z);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf("give clover total time: (without malloc free memcpy) :%.9lf sec\n ",
           double(duration) / 1e9);
  }
  {
    // free
    checkCudaErrors(cudaFree(clover));
  }
}