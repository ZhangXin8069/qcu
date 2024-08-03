#ifndef _LATTICE_BISTABCG_H
#define _LATTICE_BISTABCG_H
#include "./bistabcg.h"
#include "./dslash.h"
#include "./lattice_cuda.h"
#include "./lattice_dslash.h"
#include "define.h"
#include "lattice_complex.h"
#include <cstdlib>
#include <nccl.h>
#define PRINT_NCCL_WILSON_BISTABCG
struct LatticeBistabcg {
  LatticeSet *set_ptr;
  cudaError_t err;
  LatticeWilsonDslash dslash;
  LatticeComplex tmp0;
  LatticeComplex tmp1;
  LatticeComplex rho_prev;
  LatticeComplex rho;
  LatticeComplex alpha;
  LatticeComplex beta;
  LatticeComplex omega;
  void *ans_e, *ans_o, *x_e, *x_o, *b_e, *b_o, *b__o, *r, *r_tilde, *p, *v, *s,
      *t, *device_vec0, *device_vec1, *device_dot_vec, *device_dot_tmp_vec,
      *device_vals;
  LatticeComplex host_vals[_vals_size_];
  void _init() {
    {
      checkCudaErrors(
          cudaMallocAsync(&ans_e, set_ptr->lat_4dim_SC * sizeof(LatticeComplex),
                          set_ptr->stream));
      checkCudaErrors(
          cudaMallocAsync(&ans_o, set_ptr->lat_4dim_SC * sizeof(LatticeComplex),
                          set_ptr->stream));
      checkCudaErrors(
          cudaMallocAsync(&x_e, set_ptr->lat_4dim_SC * sizeof(LatticeComplex),
                          set_ptr->stream));
      checkCudaErrors(
          cudaMallocAsync(&x_o, set_ptr->lat_4dim_SC * sizeof(LatticeComplex),
                          set_ptr->stream));
      checkCudaErrors(
          cudaMallocAsync(&b__o, set_ptr->lat_4dim_SC * sizeof(LatticeComplex),
                          set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &r, set_ptr->lat_4dim_SC * sizeof(LatticeComplex), set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &r_tilde, set_ptr->lat_4dim_SC * sizeof(LatticeComplex),
          set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &p, set_ptr->lat_4dim_SC * sizeof(LatticeComplex), set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &v, set_ptr->lat_4dim_SC * sizeof(LatticeComplex), set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &s, set_ptr->lat_4dim_SC * sizeof(LatticeComplex), set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &t, set_ptr->lat_4dim_SC * sizeof(LatticeComplex), set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &device_vec0, set_ptr->lat_4dim_SC * sizeof(LatticeComplex),
          set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &device_vec1, set_ptr->lat_4dim_SC * sizeof(LatticeComplex),
          set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &device_dot_vec, set_ptr->lat_4dim * sizeof(LatticeComplex),
          set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &device_dot_tmp_vec, set_ptr->lat_4dim * sizeof(LatticeComplex),
          set_ptr->stream));
    }
    {
      checkCudaErrors(cudaMallocAsync(
          &device_vals, _vals_size_ * sizeof(LatticeComplex), set_ptr->stream));
      give_1zero<<<1, 1, 0, set_ptr->stream>>>(device_vals, _tmp0_);
      give_1zero<<<1, 1, 0, set_ptr->stream>>>(device_vals, _tmp1_);
      give_1one<<<1, 1, 0, set_ptr->stream>>>(device_vals, _rho_prev_);
      give_1zero<<<1, 1, 0, set_ptr->stream>>>(device_vals, _rho_);
      give_1one<<<1, 1, 0, set_ptr->stream>>>(device_vals, _alpha_);
      give_1one<<<1, 1, 0, set_ptr->stream>>>(device_vals, _omega_);
      give_1zero<<<1, 1, 0, set_ptr->stream>>>(device_vals, _send_tmp_);
      give_1zero<<<1, 1, 0, set_ptr->stream>>>(device_vals, _norm2_tmp_);
      give_1zero<<<1, 1, 0, set_ptr->stream>>>(device_vals, _diff_tmp_);
    }
    {
      give_fermi_rand<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                          set_ptr->stream>>>(x_o, 23333);
      perf_part_reduce(device_dot_vec, device_vals, device_dot_tmp_vec,
                       set_ptr->lat_4dim, set_ptr->stream);
    }
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
  void give(LatticeSet *_set_ptr) {
    set_ptr = _set_ptr;
    dslash.give(set_ptr);
    _init();
    // set_ptr->_print();
  }
  void _dslash(void *fermion_out, void *fermion_in, void *gauge) {
    // src_o-_KAPPA_**2*dslash_oe(dslash_eo(src_o))
    dslash.run_eo(device_vec0, fermion_in, gauge);
    dslash.run_oe(device_vec1, device_vec0, gauge);
    bistabcg_give_dest_o<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                           set_ptr->stream>>>(fermion_out, fermion_in,
                                              device_vec1, _KAPPA_);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
  void init(void *_b_e, void *_b_o, void *gauge) {
    b_e = _b_e;
    b_o = _b_o;
    dslash.run_oe(device_vec0, b_e, gauge);
    bistabcg_give_b__0<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                         set_ptr->stream>>>(b__o, b_o, device_vec0, _KAPPA_);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    _dslash(r, x_o, gauge);
    bistabcg_give_rr<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                       set_ptr->stream>>>(r, b__o, r_tilde);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
  void init(void *gauge) {
    checkCudaErrors(cudaMallocAsync(
        &b_e, set_ptr->lat_4dim_SC * sizeof(LatticeComplex), set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(
        &b_o, set_ptr->lat_4dim_SC * sizeof(LatticeComplex), set_ptr->stream));
    give_fermi_rand<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                        set_ptr->stream>>>(ans_e, 12138);
    give_fermi_rand<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                        set_ptr->stream>>>(ans_o, 83121);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    dslash.run_eo(device_vec0, ans_o, gauge);
    bistabcg_give_b_e<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                        set_ptr->stream>>>(b_e, ans_e, device_vec0, _KAPPA_);
    // test b=1*/
    // give_fermi_val<<<set_ptr->gridDim, set_ptr->blockDim, 0,
    //                     set_ptr->stream>>>(b_e, 1.0, 0.0);
    dslash.run_oe(device_vec1, ans_e, gauge);
    bistabcg_give_b_o<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                        set_ptr->stream>>>(b_o, ans_o, device_vec1, _KAPPA_);
    // test b=1
    // give_fermi_val<<<set_ptr->gridDim, set_ptr->blockDim, 0,
    //                     set_ptr->stream>>>(b_o, 1.0, 0.0);
    dslash.run_oe(device_vec0, b_e, gauge);
    bistabcg_give_b__0<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                         set_ptr->stream>>>(b__o, b_o, device_vec0, _KAPPA_);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    _dslash(r, x_o, gauge);
    bistabcg_give_rr<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                       set_ptr->stream>>>(r, b__o, r_tilde);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
  void dot(void *device_vec0, void *device_vec1, const int vals_index,
           const int stream_index) {
    part_dot<<<set_ptr->gridDim, set_ptr->blockDim, 0,
               set_ptr->streams[stream_index]>>>(device_vec0, device_vec1,
                                                 device_dot_vec);
    part_reduce(device_dot_vec,
                ((static_cast<LatticeComplex *>(device_vals)) + _send_tmp_),
                device_dot_tmp_vec, set_ptr->lat_4dim,
                set_ptr->streams[stream_index]);
    checkNcclErrors(ncclAllReduce(
        ((static_cast<LatticeComplex *>(device_vals)) + _send_tmp_),
        ((static_cast<LatticeComplex *>(device_vals)) + vals_index), 2,
        ncclDouble, ncclSum, set_ptr->nccl_comm,
        set_ptr->streams[stream_index]));
  }
  void diff(void *device_vec0, void *device_vec1, const int stream_index) {
    give_1zero<<<1, 1, 0, set_ptr->stream>>>(device_vals, _norm2_tmp_);
    {
      part_cut<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                 set_ptr->streams[stream_index]>>>(device_vec0, device_vec1,
                                                   device_dot_vec);
      part_reduce(device_dot_vec,
                  ((static_cast<LatticeComplex *>(device_vals)) + _send_tmp_),
                  device_dot_tmp_vec, set_ptr->lat_4dim,
                  set_ptr->streams[stream_index]);
      checkNcclErrors(ncclAllReduce(
          ((static_cast<LatticeComplex *>(device_vals)) + _send_tmp_),
          ((static_cast<LatticeComplex *>(device_vals)) + _diff_tmp_), 2,
          ncclDouble, ncclSum, set_ptr->nccl_comm,
          set_ptr->streams[stream_index]));
    }
    {
      part_dot<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                 set_ptr->streams[stream_index]>>>(device_vec1, device_vec1,
                                                   device_dot_vec);
      part_reduce(device_dot_vec,
                  ((static_cast<LatticeComplex *>(device_vals)) + _send_tmp_),
                  device_dot_tmp_vec, set_ptr->lat_4dim,
                  set_ptr->streams[stream_index]);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      checkNcclErrors(ncclAllReduce(
          ((static_cast<LatticeComplex *>(device_vals)) + _send_tmp_),
          ((static_cast<LatticeComplex *>(device_vals)) + _norm2_tmp_), 2,
          ncclDouble, ncclSum, set_ptr->nccl_comm,
          set_ptr->streams[stream_index]));
    }
    bistabcg_give_1diff<<<1, 1, 0, set_ptr->streams[stream_index]>>>(
        device_vals);
  }
  void print_vals(int loop) {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(
        cudaMemcpyAsync((static_cast<LatticeComplex *>(host_vals)),
                        (static_cast<LatticeComplex *>(device_vals)),
                        _vals_size_ * sizeof(LatticeComplex),
                        cudaMemcpyDeviceToHost, set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    std::cout << "##RANK     :" << set_ptr->node_rank << "##LOOP:" << loop
              << std::endl
              << "##tmp0     :" << host_vals[_tmp0_].real << std::endl
              << "##tmp1     :" << host_vals[_tmp1_].real << std::endl
              << "##rho_prev :" << host_vals[_rho_prev_].real << std::endl
              << "##rho      :" << host_vals[_rho_].real << std::endl
              << "##alpha    :" << host_vals[_alpha_].real << std::endl
              << "##omega    :" << host_vals[_omega_].real << std::endl
              << "##send_tmp :" << host_vals[_send_tmp_].real << std::endl
              << "##norm2_tmp:" << host_vals[_norm2_tmp_].real << std::endl
              << "##diff_tmp :" << host_vals[_diff_tmp_].real << std::endl;
  }
  void run(void *gauge) {
    dot(r, r, _norm2_tmp_, _a_);
    for (int loop = 0; loop < _MAX_ITER_; loop++) {
      dot(r_tilde, r, _rho_, _a_);
      // beta = (rho / rho_prev) * (alpha / omega);
      bistabcg_give_1beta<<<1, 1, 0, set_ptr->streams[_a_]>>>(device_vals);
      bistabcg_give_p<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                        set_ptr->streams[_a_]>>>(p, r, v, device_vals);
      {
        // break;
        checkCudaErrors(cudaMemcpyAsync(
            ((static_cast<LatticeComplex *>(host_vals)) + _norm2_tmp_),
            ((static_cast<LatticeComplex *>(device_vals)) + _norm2_tmp_),
            sizeof(LatticeComplex), cudaMemcpyDeviceToHost,
            set_ptr->streams[_a_]));
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
        std::cout << "##RANK:" << set_ptr->node_rank << "##LOOP:" << loop
                  << "##Residual:" << host_vals[_norm2_tmp_].real << std::endl;
        if ((host_vals[_norm2_tmp_].real < _TOL_ || loop == _MAX_ITER_ - 1)) {
          break;
        }
      }
      // v = A * p;
      _dslash(v, p, gauge);
      // rho_prev = rho;
      bistabcg_give_1rho_prev<<<1, 1, 0, set_ptr->streams[_b_]>>>(device_vals);
      dot(r_tilde, v, _tmp0_, _b_);
      // alpha = rho / tmp0;
      bistabcg_give_1alpha<<<1, 1, 0, set_ptr->streams[_b_]>>>(device_vals);
      bistabcg_give_s<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                        set_ptr->streams[_b_]>>>(s, r, v, device_vals);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
      // t = A * s;
      _dslash(t, s, gauge);
      dot(t, s, _tmp0_, _c_);
      dot(t, t, _tmp1_, _d_);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
      // omega = tmp0 / tmp1;
      bistabcg_give_1omega<<<1, 1, 0, set_ptr->streams[_d_]>>>(device_vals);
      bistabcg_give_x_o<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                          set_ptr->streams[_d_]>>>(x_o, p, s, device_vals);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
      bistabcg_give_r<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                        set_ptr->streams[_a_]>>>(r, s, t, device_vals);
      dot(r, r, _norm2_tmp_, _a_);
    }
  }
  void run_test(void *gauge) {
    set_ptr->_print();
    auto start = std::chrono::high_resolution_clock::now();
    run(gauge);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    set_ptr->err = cudaGetLastError();
    checkCudaErrors(set_ptr->err);
    printf(
        "nccl wilson bistabcg total time: (without malloc free memcpy) :%.9lf "
        "sec\n",
        double(duration) / 1e9);
    // checkCudaErrors(cudaDeviceSynchronize());
    diff(x_o, ans_o, _a_);
    checkCudaErrors(cudaMemcpyAsync(
        ((static_cast<LatticeComplex *>(host_vals)) + _diff_tmp_),
        ((static_cast<LatticeComplex *>(device_vals)) + _diff_tmp_),
        sizeof(LatticeComplex), cudaMemcpyDeviceToHost, set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    printf("## difference: %.16f\n", host_vals[_diff_tmp_].real);
  }
  void end() {
    checkCudaErrors(cudaFreeAsync(ans_e, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(ans_o, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(x_o, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(b__o, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(r, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(r_tilde, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(p, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(v, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(s, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(t, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(device_vec0, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(device_vec1, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(device_dot_vec, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(device_dot_tmp_vec, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(device_vals, set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
};
#endif