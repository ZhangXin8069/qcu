#pragma once

#include "qcu_complex.cuh"
#include "qcu_macro.cuh"
#include "assert.h"
#include <mpi.h>
class MPICommunicator {
private:
  int Lx_;
  int Ly_;
  int Lz_;
  int Lt_;
  int grid_front[Nd];
  int grid_back[Nd];

  void* d_partial_result_buffer;

  Complex* gauge_;
  Complex* fermion_in_;
  Complex* fermion_out_;

  Complex* d_send_front_vec[Nd];   // device pointer: 0, 1, 2, 3 represent x, y, z, t aspectively
  Complex* d_send_back_vec[Nd];    // device pointer: 0, 1, 2, 3 represent x, y, z, t aspectively
  Complex* d_recv_front_vec[Nd];   // device pointer: 0, 1, 2, 3 represent x, y, z, t aspectively
  Complex* d_recv_back_vec[Nd];    // device pointer: 0, 1, 2, 3 represent x, y, z, t aspectively

  Complex* h_send_front_vec[Nd];   // host pointer
  Complex* h_send_back_vec[Nd];    // host pointer
  Complex* h_recv_front_vec[Nd];   // host pointer
  Complex* h_recv_back_vec[Nd];    // host pointer

  Complex* gauge_shift[Nd][2];    // Nd - 4 dims    2: Front/back
  Complex* gauge_twice_shift[6][4];    // 6-combination   4: ++ +- -+ --

  Complex* h_send_gauge[Nd][2];   // Nd - 4 dims    2: Front/back
  Complex* d_send_gauge[Nd][2];   // Nd - 4 dims    2: Front/back
  Complex* h_recv_gauge[Nd][2];   // Nd - 4 dims    2: Front/back
  Complex* d_recv_gauge[Nd][2];   // Nd - 4 dims    2: Front/back

  MPI_Request send_front_req[Nd];
  MPI_Request send_back_req[Nd];
  MPI_Request recv_front_req[Nd];
  MPI_Request recv_back_req[Nd];

  MPI_Status send_front_status[Nd];
  MPI_Status send_back_status[Nd];
  MPI_Status recv_front_status[Nd];
  MPI_Status recv_back_status[Nd];
public:
  Complex* getOriginGauge() const {
    return gauge_;
  }
  Complex** getShiftGauge() {
    return &(gauge_shift[0][0]);
  }
  Complex** getShiftShiftGauge() {
    return &(gauge_twice_shift[0][0]);
  }

  MPICommunicator (Complex* gauge, Complex* fermion_in, Complex* fermion_out, int Lx, int Ly, int Lz, int Lt) : gauge_(gauge), fermion_in_(fermion_in), fermion_out_(fermion_out), Lx_(Lx), Ly_(Ly), Lz_(Lz), Lt_(Lt) {
    for (int i = 0; i < Nd; i++) {
      d_send_front_vec[i] = nullptr;
      d_send_back_vec[i] = nullptr;
      d_recv_front_vec[i] = nullptr;
      d_recv_back_vec[i] = nullptr;

      h_send_front_vec[i] = nullptr;
      h_send_back_vec[i] = nullptr;
      h_recv_front_vec[i] = nullptr;
      h_recv_back_vec[i] = nullptr;
    }
    allocateBuffer();
    calculateAdjacentProcess();
    // allocateGaugeBuffer();
    // prepareGauge();
  }
  ~MPICommunicator() {
    // freeBuffer();
    // freeGaugeBuffer();
  }
  MPICommunicator (void* gauge, void* fermion_in, void* fermion_out, int Lx, int Ly, int Lz, int Lt) : gauge_(static_cast<Complex*>(gauge)), fermion_in_(static_cast<Complex*>(fermion_in)), fermion_out_(static_cast<Complex*>(fermion_out)), Lx_(Lx), Ly_(Ly), Lz_(Lz), Lt_(Lt){
    for (int i = 0; i < Nd; i++) {
      d_send_front_vec[i] = nullptr;
      d_send_back_vec[i] = nullptr;
      d_recv_front_vec[i] = nullptr;
      d_recv_back_vec[i] = nullptr;
      h_send_front_vec[i] = nullptr;
      h_send_back_vec[i] = nullptr;
      h_recv_front_vec[i] = nullptr;
      h_recv_back_vec[i] = nullptr;
    }
    allocateBuffer();
    calculateAdjacentProcess();
    // allocateGaugeBuffer();
    // prepareGauge();
  }



  int getAdjacentProcess (int front_back, int direction) const{
    assert(front_back==FRONT || front_back==BACK);
    assert(direction==X_DIRECTION || direction==Y_DIRECTION || direction==Z_DIRECTION || direction==T_DIRECTION);
    if (front_back == FRONT) {
      return grid_front[direction];
    } else {
      return grid_back[direction];
    }
  }
  // return device pointer
  Complex* getSendBufferAddr(int front_back, int direction) const { // TO SEND
    assert(front_back==FRONT || front_back==BACK);
    assert(direction==X_DIRECTION || direction==Y_DIRECTION || direction==Z_DIRECTION || direction==T_DIRECTION);
    if (front_back==FRONT) {
      return d_send_front_vec[direction];
    }
    else {
      return d_send_back_vec[direction];
    }
  }
  Complex* getHostSendBufferAddr(int front_back, int direction) const { // TO SEND
    assert(front_back==FRONT || front_back==BACK);
    assert(direction==X_DIRECTION || direction==Y_DIRECTION || direction==Z_DIRECTION || direction==T_DIRECTION);
    if (front_back==FRONT) {
      return h_send_front_vec[direction];
    }
    else {
      return h_send_back_vec[direction];
    }
  }
  Complex* getRecvBufferAddr(int front_back, int direction) const { // TO RECEIVE
    assert(front_back==FRONT || front_back==BACK);
    assert(direction==X_DIRECTION || direction==Y_DIRECTION || direction==Z_DIRECTION || direction==T_DIRECTION);
    if (front_back==FRONT) {
      return d_recv_front_vec[direction];
    }
    else {
      return d_recv_back_vec[direction];
    }
  }
  Complex* getHostRecvBufferAddr(int front_back, int direction) const{ // TO RECEIVE
    assert(front_back==FRONT || front_back==BACK);
    assert(direction==X_DIRECTION || direction==Y_DIRECTION || direction==Z_DIRECTION || direction==T_DIRECTION);
    if (front_back==FRONT) {
      return h_recv_front_vec[direction];
    }
    else {
      return h_recv_back_vec[direction];
    }
  }

  void preDslash(void* fermion_in, int parity, int invert_flag);
  void postDslash(void* fermion_out, int parity, int invert_flag);
  void recvBoundaryVector(int direction);
  void prepareFrontBoundaryVector(void* fermion_in, int direction, int parity, int invert_flag);
  void prepareGauge();
  void shiftGauge(void* src_gauge, void* front_shift_gauge, void* back_shift_gauge, int direction);
  void shiftGaugeKernel(void* src_gauge, void* front_shift_gauge, void* back_shift_gauge, int direction);

  void prepareBoundaryGauge(void* src_gauge, int direction);

  void qcuGaugeMPIBarrier(int direction);
  void recvGauge(int direction);
  void sendGauge(int direction);

  void allocateGaugeBuffer(); // gauge
  void allocateBuffer();  // vector
  
  void calculateAdjacentProcess();

  void interprocess_saxpy_barrier(void* x, void* y, void* scalar, int vol);
  void interprocess_inner_prod_barrier(void* x, void* y, void* result, int vol);
  void interprocess_sax_barrier (void* x, void* scalar, int vol);
  void freeBuffer();
  void freeGaugeBuffer();
};