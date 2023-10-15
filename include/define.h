#pragma optimize(5)
#include <cstdio>
#include <random>
#define BLOCK_SIZE 256
#define FORWARD 1
#define NONEWARD 0
#define BACKWARD -1
#define LAT_C 3
#define LAT_S 4
#define LAT_D 4
#define X 0
#define Y 1
#define Z 2
#define T 3

// #define DEBUG

#ifdef DEBUG
#define NOTE 666
cout << "\033[31m" << str << "\033[0m" << endl;
cout << "\033[32m" << str << "\033[0m" << endl;
#endif
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
// little strange, but don't want change
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

#define add_value(U, tmp, n)                                                   \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] += tmp;                                                             \
    }                                                                          \
  }

#define subt_value(U, tmp, n)                                                  \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] -= tmp;                                                             \
    }                                                                          \
  }

#define mult_value(U, tmp, n)                                                  \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] *= tmp;                                                             \
    }                                                                          \
  }

#define divi_value(U, tmp, n)                                                  \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] /= tmp;                                                             \
    }                                                                          \
  }

#define add_ptr(U, tmp, n)                                                     \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] += tmp[i];                                                          \
    }                                                                          \
  }

#define subt_ptr(U, tmp, n)                                                    \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] -= tmp[i];                                                          \
    }                                                                          \
  }

#define mult_ptr(U, tmp, n)                                                    \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] *= tmp[i];                                                          \
    }                                                                          \
  }

#define divi_ptr(U, tmp, n)                                                    \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] /= tmp[i];                                                          \
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

#define print_tmp(input_tmp, n)                                                \
  {                                                                            \
    register LatticeComplex *tmp_vec =                                         \
        static_cast<LatticeComplex *>(input_tmp);                              \
    printf("############\n");                                                  \
    for (int i = 0; i < n; i++) {                                              \
      printf("[%.9lf,%.9lf]\n", tmp_vec[i].real, tmp_vec[i].imag);             \
    }                                                                          \
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

#define move_backward(move, y, lat_y)                                          \
  { move = -1 + (y == 0) * lat_y; }

#define move_forward(move, y, lat_y)                                           \
  { move = 1 - (y == lat_y - 1) * lat_y; }

#define move_backward_x(move, x, lat_x, eo, parity)                            \
  { move = (-1 + (x == 0) * lat_x) * (eo == parity); }

#define move_forward_x(move, x, lat_x, eo, parity)                             \
  { move = (1 - (x == lat_x - 1) * lat_x) * (eo != parity); }


#ifdef JOD557
#define make_clover_f(move, x, lat_x
,lat_y
,lat_z
,lat_t, eo, parity)
{
  // XY
  give_value(U, zero, 9);
  {
    //// x,y,z,t;x
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x+1,y,z,t;y
    move_forward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 2 + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z,t;x;dag
    move_forward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_xcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, 9);
  {
    //// x,y,z,t;y
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y+1,z,t;x;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    move_forward(move1, y, lat_y);
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;y;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 2 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, 9);
  {
    //// x-1,y,z,t;x;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y-1,z,t;y;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, y, lat_y);
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y-1,z,t;x
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, y, lat_y);
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, 9);
  {
    //// x,y-1,z,t;y;dag
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z,t;x
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_xcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y-1,z,t;y
    move_forward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, y, lat_y);
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, 9);
}

#endif