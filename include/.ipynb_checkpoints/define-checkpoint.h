#include <cstdio>
#include <random>
#define BLOCK_SIZE 256
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
