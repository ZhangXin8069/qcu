// clang-format off
#include "../include/qcu.h"
#ifdef CLOVER_DSLASH
// wait for rebuild
// clang-format on
__global__ void pick_up_u_x(void *device_U, void *device_lat_xyzt,
                            int node_rank, int device_flag,
                            void *device_u_b_x_send_vec,
                            void *device_u_f_x_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *lat_xyzt = static_cast<int *>(device_lat_xyzt);
  int lat_x = 1;
  int lat_y = lat_xyzt[_Y_];
  int lat_z = lat_xyzt[_Z_];
  // int lat_t = lat_xyzt[_T_];
  int lat_tzyx = lat_xyzt[_XYZT_];
  int tmp1;
  tmp1 = lat_x * lat_y * lat_z;
  int t = tmp0 / tmp1;
  tmp0 -= t * tmp1;
  tmp1 = lat_x * lat_y;
  int z = tmp0 / tmp1;
  tmp0 -= z * tmp1;
  int y = tmp0 / lat_x;
  // int x = tmp0 - y * lat_x;
  lat_x = lat_xyzt[_X_];
  LatticeComplex *origin_U = static_cast<LatticeComplex *>(device_U);
  LatticeComplex *tmp_U;
  // u_1dim_send_vec
  //// x
  LatticeComplex *u_b_x_send_vec =
      (static_cast<LatticeComplex *>(device_u_b_x_send_vec) + idx);
  LatticeComplex *u_f_x_send_vec =
      (static_cast<LatticeComplex *>(device_u_f_x_send_vec) + idx);
  // b_x
  tmp_U = (origin_U + ((((t)*lat_z + z) * lat_y + y) * lat_x + 0));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_b_x_send_vec[i * lat_tzyx / lat_x] = tmp_U[i * lat_tzyx];
  }
  // f_x
  tmp_U = (origin_U + ((((t)*lat_z + z) * lat_y + y) * lat_x + lat_x - 1));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_f_x_send_vec[i * lat_tzyx / lat_x] = tmp_U[i * lat_tzyx];
  }
}
__global__ void pick_up_u_y(void *device_U, void *device_lat_xyzt,
                            int node_rank, int device_flag,
                            void *device_u_b_y_send_vec,
                            void *device_u_f_y_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *lat_xyzt = static_cast<int *>(device_lat_xyzt);
  int lat_x = lat_xyzt[_X_];
  int lat_y = 1;
  int lat_z = lat_xyzt[_Z_];
  // int lat_t = lat_xyzt[_T_];
  int lat_tzyx = lat_xyzt[_XYZT_];
  int tmp1;
  tmp1 = lat_x * lat_y * lat_z;
  int t = tmp0 / tmp1;
  tmp0 -= t * tmp1;
  tmp1 = lat_x * lat_y;
  int z = tmp0 / tmp1;
  tmp0 -= z * tmp1;
  int y = tmp0 / lat_x;
  int x = tmp0 - y * lat_x;
  lat_y = lat_xyzt[_Y_];
  LatticeComplex *origin_U = static_cast<LatticeComplex *>(device_U);
  LatticeComplex *tmp_U;
  // u_1dim_send_vec
  //// y
  LatticeComplex *u_b_y_send_vec =
      (static_cast<LatticeComplex *>(device_u_b_y_send_vec) + idx);
  LatticeComplex *u_f_y_send_vec =
      (static_cast<LatticeComplex *>(device_u_f_y_send_vec) + idx);
  // b_y
  tmp_U = (origin_U + ((((t)*lat_z + z) * lat_y + 0) * lat_x + x));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_b_y_send_vec[i * lat_tzyx / lat_y] = tmp_U[i * lat_tzyx];
  }
  // f_y
  tmp_U = (origin_U + ((((t)*lat_z + z) * lat_y + lat_y - 1) * lat_x + x));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_f_y_send_vec[i * lat_tzyx / lat_y] = tmp_U[i * lat_tzyx];
  }
}
__global__ void pick_up_u_z(void *device_U, void *device_lat_xyzt,
                            int node_rank, int device_flag,
                            void *device_u_b_z_send_vec,
                            void *device_u_f_z_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *lat_xyzt = static_cast<int *>(device_lat_xyzt);
  int lat_x = lat_xyzt[_X_];
  int lat_y = lat_xyzt[_Y_];
  int lat_z = 1;
  // int lat_t = lat_xyzt[_T_];
  int lat_tzyx = lat_xyzt[_XYZT_];
  int tmp1;
  tmp1 = lat_x * lat_y * lat_z;
  int t = tmp0 / tmp1;
  tmp0 -= t * tmp1;
  tmp1 = lat_x * lat_y;
  int z = tmp0 / tmp1;
  tmp0 -= z * tmp1;
  int y = tmp0 / lat_x;
  int x = tmp0 - y * lat_x;
  lat_z = lat_xyzt[_Z_];
  LatticeComplex *origin_U = static_cast<LatticeComplex *>(device_U);
  LatticeComplex *tmp_U;
  // u_1dim_send_vec
  //// z
  LatticeComplex *u_b_z_send_vec =
      (static_cast<LatticeComplex *>(device_u_b_z_send_vec) + idx);
  LatticeComplex *u_f_z_send_vec =
      (static_cast<LatticeComplex *>(device_u_f_z_send_vec) + idx);
  // b_z
  tmp_U = (origin_U + ((((t)*lat_z + 0) * lat_y + y) * lat_x + x));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_b_z_send_vec[i * lat_tzyx / lat_z] = tmp_U[i * lat_tzyx];
  }
  // f_z
  tmp_U = (origin_U + ((((t)*lat_z + lat_z - 1) * lat_y + y) * lat_x + x));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_f_z_send_vec[i * lat_tzyx / lat_z] = tmp_U[i * lat_tzyx];
  }
}
__global__ void pick_up_u_t(void *device_U, void *device_lat_xyzt,
                            int node_rank, int device_flag,
                            void *device_u_b_t_send_vec,
                            void *device_u_f_t_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *lat_xyzt = static_cast<int *>(device_lat_xyzt);
  int lat_x = lat_xyzt[_X_];
  int lat_y = lat_xyzt[_Y_];
  int lat_z = lat_xyzt[_Z_];
  int lat_t = 1;
  int lat_tzyx = lat_xyzt[_XYZT_];
  int tmp1;
  tmp1 = lat_x * lat_y * lat_z;
  int t = tmp0 / tmp1;
  tmp0 -= t * tmp1;
  tmp1 = lat_x * lat_y;
  int z = tmp0 / tmp1;
  tmp0 -= z * tmp1;
  int y = tmp0 / lat_x;
  int x = tmp0 - y * lat_x;
  lat_t = lat_xyzt[_T_];
  LatticeComplex *origin_U = static_cast<LatticeComplex *>(device_U);
  LatticeComplex *tmp_U;
  // u_1dim_send_vec
  //// t
  LatticeComplex *u_b_t_send_vec =
      (static_cast<LatticeComplex *>(device_u_b_t_send_vec) + idx);
  LatticeComplex *u_f_t_send_vec =
      (static_cast<LatticeComplex *>(device_u_f_t_send_vec) + idx);
  // b_t
  tmp_U = (origin_U + ((((0) * lat_z + z) * lat_y + y) * lat_x + x));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_b_t_send_vec[i * lat_tzyx / lat_t] = tmp_U[i * lat_tzyx];
  }
  // f_t
  tmp_U = (origin_U + ((((lat_t - 1) * lat_z + z) * lat_y + y) * lat_x + x));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_f_t_send_vec[i * lat_tzyx / lat_t] = tmp_U[i * lat_tzyx];
  }
}
__global__ void pick_up_u_xy(void *device_U, void *device_lat_xyzt,
                             int node_rank, int device_flag,
                             void *device_u_b_x_b_y_send_vec,
                             void *device_u_f_x_b_y_send_vec,
                             void *device_u_b_x_f_y_send_vec,
                             void *device_u_f_x_f_y_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *lat_xyzt = static_cast<int *>(device_lat_xyzt);
  int lat_x = 1;
  int lat_y = 1;
  int lat_z = lat_xyzt[_Z_];
  // int lat_t = lat_xyzt[_T_];
  int lat_tzyx = lat_xyzt[_XYZT_];
  int tmp1;
  tmp1 = lat_x * lat_y * lat_z;
  int t = tmp0 / tmp1;
  tmp0 -= t * tmp1;
  tmp1 = lat_x * lat_y;
  int z = tmp0 / tmp1;
  tmp0 -= z * tmp1;
  // int y = tmp0 / lat_x;
  // int x = tmp0 - y * lat_x;
  lat_x = lat_xyzt[_X_];
  lat_y = lat_xyzt[_Y_];
  LatticeComplex *origin_U = static_cast<LatticeComplex *>(device_U);
  LatticeComplex *tmp_U;
  // u_2dim_send_vec
  //// xy
  LatticeComplex *u_b_x_b_y_send_vec =
      (static_cast<LatticeComplex *>(device_u_b_x_b_y_send_vec) + idx);
  LatticeComplex *u_f_x_b_y_send_vec =
      (static_cast<LatticeComplex *>(device_u_f_x_b_y_send_vec) + idx);
  LatticeComplex *u_b_x_f_y_send_vec =
      (static_cast<LatticeComplex *>(device_u_b_x_f_y_send_vec) + idx);
  LatticeComplex *u_f_x_f_y_send_vec =
      (static_cast<LatticeComplex *>(device_u_f_x_f_y_send_vec) + idx);
  // b_x_b_y
  tmp_U = (origin_U + ((((t)*lat_z + z) * lat_y + 0) * lat_x + 0));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_b_x_b_y_send_vec[i * lat_tzyx / lat_x / lat_y] = tmp_U[i * lat_tzyx];
  }
  // f_x_b_y
  tmp_U = (origin_U + ((((t)*lat_z + z) * lat_y + 0) * lat_x + lat_x - 1));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_f_x_b_y_send_vec[i * lat_tzyx / lat_x / lat_y] = tmp_U[i * lat_tzyx];
  }
  // b_x_f_y
  tmp_U = (origin_U + ((((t)*lat_z + z) * lat_y + lat_y - 1) * lat_x + 0));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_b_x_f_y_send_vec[i * lat_tzyx / lat_x / lat_y] = tmp_U[i * lat_tzyx];
  }
  // f_x_f_y
  tmp_U =
      (origin_U + ((((t)*lat_z + z) * lat_y + lat_y - 1) * lat_x + lat_x - 1));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_f_x_f_y_send_vec[i * lat_tzyx / lat_x / lat_y] = tmp_U[i * lat_tzyx];
  }
}
__global__ void pick_up_u_xz(void *device_U, void *device_lat_xyzt,
                             int node_rank, int device_flag,
                             void *device_u_b_x_b_z_send_vec,
                             void *device_u_f_x_b_z_send_vec,
                             void *device_u_b_x_f_z_send_vec,
                             void *device_u_f_x_f_z_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *lat_xyzt = static_cast<int *>(device_lat_xyzt);
  int lat_x = 1;
  int lat_y = lat_xyzt[_Y_];
  int lat_z = 1;
  // int lat_t = lat_xyzt[_T_];
  int lat_tzyx = lat_xyzt[_XYZT_];
  int tmp1;
  tmp1 = lat_x * lat_y * lat_z;
  int t = tmp0 / tmp1;
  tmp0 -= t * tmp1;
  tmp1 = lat_x * lat_y;
  int z = tmp0 / tmp1;
  tmp0 -= z * tmp1;
  int y = tmp0 / lat_x;
  // int x = tmp0 - y * lat_x;
  lat_x = lat_xyzt[_X_];
  lat_z = lat_xyzt[_Z_];
  LatticeComplex *origin_U = static_cast<LatticeComplex *>(device_U);
  LatticeComplex *tmp_U;
  // u_2dim_send_vec
  // xz
  LatticeComplex *u_b_x_b_z_send_vec =
      (static_cast<LatticeComplex *>(device_u_b_x_b_z_send_vec) + idx);
  LatticeComplex *u_f_x_b_z_send_vec =
      (static_cast<LatticeComplex *>(device_u_f_x_b_z_send_vec) + idx);
  LatticeComplex *u_b_x_f_z_send_vec =
      (static_cast<LatticeComplex *>(device_u_b_x_f_z_send_vec) + idx);
  LatticeComplex *u_f_x_f_z_send_vec =
      (static_cast<LatticeComplex *>(device_u_f_x_f_z_send_vec) + idx);
  // b_x_b_z
  tmp_U = (origin_U + ((((t)*lat_z + 0) * lat_y + y) * lat_x + 0));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_b_x_b_z_send_vec[i * lat_tzyx / lat_x / lat_z] = tmp_U[i * lat_tzyx];
  }
  // f_x_b_z
  tmp_U = (origin_U + ((((t)*lat_z + 0) * lat_y + y) * lat_x + lat_x - 1));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_f_x_b_z_send_vec[i * lat_tzyx / lat_x / lat_z] = tmp_U[i * lat_tzyx];
  }
  // b_x_f_z
  tmp_U = (origin_U + ((((t)*lat_z + lat_z - 1) * lat_y + y) * lat_x + 0));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_b_x_f_z_send_vec[i * lat_tzyx / lat_x / lat_z] = tmp_U[i * lat_tzyx];
  }
  // f_x_f_z
  tmp_U =
      (origin_U + ((((t)*lat_z + lat_z - 1) * lat_y + y) * lat_x + lat_x - 1));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_f_x_f_z_send_vec[i * lat_tzyx / lat_x / lat_z] = tmp_U[i * lat_tzyx];
  }
}
__global__ void pick_up_u_xt(void *device_U, void *device_lat_xyzt,
                             int node_rank, int device_flag,
                             void *device_u_b_x_b_t_send_vec,
                             void *device_u_f_x_b_t_send_vec,
                             void *device_u_b_x_f_t_send_vec,
                             void *device_u_f_x_f_t_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *lat_xyzt = static_cast<int *>(device_lat_xyzt);
  int lat_x = 1;
  int lat_y = lat_xyzt[_Y_];
  int lat_z = lat_xyzt[_Z_];
  int lat_t = 1;
  int lat_tzyx = lat_xyzt[_XYZT_];
  int tmp1;
  tmp1 = lat_x * lat_y * lat_z;
  int t = tmp0 / tmp1;
  tmp0 -= t * tmp1;
  tmp1 = lat_x * lat_y;
  int z = tmp0 / tmp1;
  tmp0 -= z * tmp1;
  int y = tmp0 / lat_x;
  // int x = tmp0 - y * lat_x;
  lat_x = lat_xyzt[_X_];
  lat_t = lat_xyzt[_T_];
  LatticeComplex *origin_U = static_cast<LatticeComplex *>(device_U);
  LatticeComplex *tmp_U;
  // u_2dim_send_vec
  // xt
  LatticeComplex *u_b_x_b_t_send_vec =
      (static_cast<LatticeComplex *>(device_u_b_x_b_t_send_vec) + idx);
  LatticeComplex *u_f_x_b_t_send_vec =
      (static_cast<LatticeComplex *>(device_u_f_x_b_t_send_vec) + idx);
  LatticeComplex *u_b_x_f_t_send_vec =
      (static_cast<LatticeComplex *>(device_u_b_x_f_t_send_vec) + idx);
  LatticeComplex *u_f_x_f_t_send_vec =
      (static_cast<LatticeComplex *>(device_u_f_x_f_t_send_vec) + idx);
  // b_x_b_t
  tmp_U = (origin_U + ((((0) * lat_z + z) * lat_y + y) * lat_x + 0));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_b_x_b_t_send_vec[i * lat_tzyx / lat_x / lat_t] = tmp_U[i * lat_tzyx];
  }
  // f_x_b_t
  tmp_U = (origin_U + ((((0) * lat_z + z) * lat_y + y) * lat_x + lat_x - 1));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_f_x_b_t_send_vec[i * lat_tzyx / lat_x / lat_t] = tmp_U[i * lat_tzyx];
  }
  // b_x_f_t
  tmp_U = (origin_U + ((((lat_t - 1) * lat_z + z) * lat_y + y) * lat_x + 0));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_b_x_f_t_send_vec[i * lat_tzyx / lat_x / lat_t] = tmp_U[i * lat_tzyx];
  }
  // f_x_f_t
  tmp_U = (origin_U +
           ((((lat_t - 1) * lat_z + z) * lat_y + y) * lat_x + lat_x - 1));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_f_x_f_t_send_vec[i * lat_tzyx / lat_x / lat_t] = tmp_U[i * lat_tzyx];
  }
}
__global__ void pick_up_u_yz(void *device_U, void *device_lat_xyzt,
                             int node_rank, int device_flag,
                             void *device_u_b_y_b_z_send_vec,
                             void *device_u_f_y_b_z_send_vec,
                             void *device_u_b_y_f_z_send_vec,
                             void *device_u_f_y_f_z_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *lat_xyzt = static_cast<int *>(device_lat_xyzt);
  int lat_x = lat_xyzt[_X_];
  int lat_y = 1;
  int lat_z = 1;
  // int lat_t = lat_xyzt[_T_];
  int lat_tzyx = lat_xyzt[_XYZT_];
  int tmp1;
  tmp1 = lat_x * lat_y * lat_z;
  int t = tmp0 / tmp1;
  tmp0 -= t * tmp1;
  tmp1 = lat_x * lat_y;
  int z = tmp0 / tmp1;
  tmp0 -= z * tmp1;
  int y = tmp0 / lat_x;
  int x = tmp0 - y * lat_x;
  lat_y = lat_xyzt[_Y_];
  lat_z = lat_xyzt[_Z_];
  LatticeComplex *origin_U = static_cast<LatticeComplex *>(device_U);
  LatticeComplex *tmp_U;
  // u_2dim_send_vec
  // yz
  LatticeComplex *u_b_y_b_z_send_vec =
      (static_cast<LatticeComplex *>(device_u_b_y_b_z_send_vec) + idx);
  LatticeComplex *u_f_y_b_z_send_vec =
      (static_cast<LatticeComplex *>(device_u_f_y_b_z_send_vec) + idx);
  LatticeComplex *u_b_y_f_z_send_vec =
      (static_cast<LatticeComplex *>(device_u_b_y_f_z_send_vec) + idx);
  LatticeComplex *u_f_y_f_z_send_vec =
      (static_cast<LatticeComplex *>(device_u_f_y_f_z_send_vec) + idx);
  // b_y_b_z
  tmp_U = (origin_U + ((((t)*lat_z + 0) * lat_y + 0) * lat_x + x));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_b_y_b_z_send_vec[i * lat_tzyx / lat_y / lat_z] = tmp_U[i * lat_tzyx];
  }
  // f_y_b_z
  tmp_U = (origin_U + ((((t)*lat_z + 0) * lat_y + lat_y - 1) * lat_x + x));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_f_y_b_z_send_vec[i * lat_tzyx / lat_y / lat_z] = tmp_U[i * lat_tzyx];
  }
  // b_y_f_z
  tmp_U = (origin_U + ((((t)*lat_z + lat_z - 1) * lat_y + 0) * lat_x + x));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_b_y_f_z_send_vec[i * lat_tzyx / lat_y / lat_z] = tmp_U[i * lat_tzyx];
  }
  // f_y_f_z
  tmp_U =
      (origin_U + ((((t)*lat_z + lat_z - 1) * lat_y + lat_y - 1) * lat_x + x));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_f_y_f_z_send_vec[i * lat_tzyx / lat_y / lat_z] = tmp_U[i * lat_tzyx];
  }
}
__global__ void pick_up_u_yt(void *device_U, void *device_lat_xyzt,
                             int node_rank, int device_flag,
                             void *device_u_b_y_b_t_send_vec,
                             void *device_u_f_y_b_t_send_vec,
                             void *device_u_b_y_f_t_send_vec,
                             void *device_u_f_y_f_t_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *lat_xyzt = static_cast<int *>(device_lat_xyzt);
  int lat_x = lat_xyzt[_X_];
  int lat_y = 1;
  int lat_z = lat_xyzt[_Z_];
  int lat_t = 1;
  int lat_tzyx = lat_xyzt[_XYZT_];
  int tmp1;
  tmp1 = lat_x * lat_y * lat_z;
  int t = tmp0 / tmp1;
  tmp0 -= t * tmp1;
  tmp1 = lat_x * lat_y;
  int z = tmp0 / tmp1;
  tmp0 -= z * tmp1;
  int y = tmp0 / lat_x;
  int x = tmp0 - y * lat_x;
  lat_y = lat_xyzt[_Y_];
  lat_t = lat_xyzt[_T_];
  LatticeComplex *origin_U = static_cast<LatticeComplex *>(device_U);
  LatticeComplex *tmp_U;
  // u_2dim_send_vec
  // yt
  LatticeComplex *u_b_y_b_t_send_vec =
      (static_cast<LatticeComplex *>(device_u_b_y_b_t_send_vec) + idx);
  LatticeComplex *u_f_y_b_t_send_vec =
      (static_cast<LatticeComplex *>(device_u_f_y_b_t_send_vec) + idx);
  LatticeComplex *u_b_y_f_t_send_vec =
      (static_cast<LatticeComplex *>(device_u_b_y_f_t_send_vec) + idx);
  LatticeComplex *u_f_y_f_t_send_vec =
      (static_cast<LatticeComplex *>(device_u_f_y_f_t_send_vec) + idx);
  // b_y_b_t
  tmp_U = (origin_U + ((((0) * lat_z + z) * lat_y + 0) * lat_x + x));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_b_y_b_t_send_vec[i * lat_tzyx / lat_y / lat_t] = tmp_U[i * lat_tzyx];
  }
  // f_y_b_t
  tmp_U = (origin_U + ((((0) * lat_z + z) * lat_y + lat_y - 1) * lat_x + x));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_f_y_b_t_send_vec[i * lat_tzyx / lat_y / lat_t] = tmp_U[i * lat_tzyx];
  }
  // b_y_f_t
  tmp_U = (origin_U + ((((lat_t - 1) * lat_z + z) * lat_y + 0) * lat_x + x));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_b_y_f_t_send_vec[i * lat_tzyx / lat_y / lat_t] = tmp_U[i * lat_tzyx];
  }
  // f_y_f_t
  tmp_U = (origin_U +
           ((((lat_t - 1) * lat_z + z) * lat_y + lat_y - 1) * lat_x + x));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_f_y_f_t_send_vec[i * lat_tzyx / lat_y / lat_t] = tmp_U[i * lat_tzyx];
  }
}
__global__ void pick_up_u_zt(void *device_U, void *device_lat_xyzt,
                             int node_rank, int device_flag,
                             void *device_u_b_z_b_t_send_vec,
                             void *device_u_f_z_b_t_send_vec,
                             void *device_u_b_z_f_t_send_vec,
                             void *device_u_f_z_f_t_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *lat_xyzt = static_cast<int *>(device_lat_xyzt);
  int lat_x = lat_xyzt[_X_];
  int lat_y = lat_xyzt[_Y_];
  int lat_z = 1;
  int lat_t = 1;
  int lat_tzyx = lat_xyzt[_XYZT_];
  int tmp1;
  tmp1 = lat_x * lat_y * lat_z;
  int t = tmp0 / tmp1;
  tmp0 -= t * tmp1;
  tmp1 = lat_x * lat_y;
  int z = tmp0 / tmp1;
  tmp0 -= z * tmp1;
  int y = tmp0 / lat_x;
  int x = tmp0 - y * lat_x;
  lat_z = lat_xyzt[_Z_];
  lat_t = lat_xyzt[_T_];
  LatticeComplex *origin_U = static_cast<LatticeComplex *>(device_U);
  LatticeComplex *tmp_U;
  // u_2dim_send_vec
  LatticeComplex *u_b_z_b_t_send_vec =
      (static_cast<LatticeComplex *>(device_u_b_z_b_t_send_vec) + idx);
  LatticeComplex *u_f_z_b_t_send_vec =
      (static_cast<LatticeComplex *>(device_u_f_z_b_t_send_vec) + idx);
  LatticeComplex *u_b_z_f_t_send_vec =
      (static_cast<LatticeComplex *>(device_u_b_z_f_t_send_vec) + idx);
  LatticeComplex *u_f_z_f_t_send_vec =
      (static_cast<LatticeComplex *>(device_u_f_z_f_t_send_vec) + idx);
  // b_z_b_t
  tmp_U = (origin_U + ((((0) * lat_z + 0) * lat_y + y) * lat_x + x));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_b_z_b_t_send_vec[i * lat_tzyx / lat_z / lat_t] = tmp_U[i * lat_tzyx];
  }
  // f_z_b_t
  tmp_U = (origin_U + ((((0) * lat_z + lat_z - 1) * lat_y + y) * lat_x + x));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_f_z_b_t_send_vec[i * lat_tzyx / lat_z / lat_t] = tmp_U[i * lat_tzyx];
  }
  // b_z_f_t
  tmp_U = (origin_U + ((((lat_t - 1) * lat_z + 0) * lat_y + y) * lat_x + x));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_b_z_f_t_send_vec[i * lat_tzyx / lat_z / lat_t] = tmp_U[i * lat_tzyx];
  }
  // f_z_f_t
  tmp_U = (origin_U +
           ((((lat_t - 1) * lat_z + lat_z - 1) * lat_y + y) * lat_x + x));
  for (int i = 0; i < _LAT_PDCC_; i++) {
    u_f_z_f_t_send_vec[i * lat_tzyx / lat_z / lat_t] = tmp_U[i * lat_tzyx];
  }
}
#endif