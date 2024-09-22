#ifndef _LATTICE_BISTABCG_H
#define _LATTICE_BISTABCG_H
// clang-format off
#include "./bistabcg.h"
#include "./lattice_cuda.h"
#include "./lattice_wilson_dslash.h"
// clang-format on
#define PRINT_NCCL_WILSON_BISTABCG
struct LatticeBistabcg {
  LatticeSet *set_ptr;
  cudaError_t err;
  LatticeWilsonDslash wilson_dslash;
  LatticeComplex tmp0;
  LatticeComplex tmp1;
  LatticeComplex rho_prev;
  LatticeComplex rho;
  LatticeComplex alpha;
  LatticeComplex beta;
  LatticeComplex omega;
  void *gauge, *ans_e, *ans_o, *x_e, *x_o, *b_e, *b_o, *b__o, *r, *r_tilde, *p,
      *v, *s, *t, *device_vec0, *device_vec1, *device_vals;
  LatticeComplex host_vals[_vals_size_];
  int if_input, if_test;
  void _init() {
    {
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
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
      give_1custom<<<1, 1, 0, set_ptr->stream>>>(
          device_vals, _lat_xyzt_, double(set_ptr->lat_4dim), 0.0);
    }
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
  void __init() {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    if (if_input == 0) {
      checkCudaErrors(
          cudaMallocAsync(&x_o, set_ptr->lat_4dim_SC * sizeof(LatticeComplex),
                          set_ptr->stream));
      checkCudaErrors(
          cudaMallocAsync(&ans_e, set_ptr->lat_4dim_SC * sizeof(LatticeComplex),
                          set_ptr->stream));
      checkCudaErrors(
          cudaMallocAsync(&ans_o, set_ptr->lat_4dim_SC * sizeof(LatticeComplex),
                          set_ptr->stream));
      give_random_vals<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                         set_ptr->stream>>>(ans_e, 12138);
      give_random_vals<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                         set_ptr->stream>>>(ans_o, 83121);
      checkCudaErrors(
          cudaMallocAsync(&b_e, set_ptr->lat_4dim_SC * sizeof(LatticeComplex),
                          set_ptr->stream));
      checkCudaErrors(
          cudaMallocAsync(&b_o, set_ptr->lat_4dim_SC * sizeof(LatticeComplex),
                          set_ptr->stream));
      wilson_dslash.run_eo(device_vec0, ans_o, gauge);
      bistabcg_give_b_e<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                          set_ptr->stream>>>(b_e, ans_e, device_vec0, _KAPPA_,
                                             device_vals);
      wilson_dslash.run_oe(device_vec1, ans_e, gauge);
      bistabcg_give_b_o<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                          set_ptr->stream>>>(b_o, ans_o, device_vec1, _KAPPA_,
                                             device_vals);
    }
    { // give b__0, x_o, rr
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      wilson_dslash.run_oe(device_vec0, b_e, gauge);
      bistabcg_give_b__0<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                           set_ptr->stream>>>(b__o, b_o, device_vec0, _KAPPA_,
                                              device_vals);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      give_random_vals<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                         set_ptr->stream>>>(x_o, 23333);
      _wilson_dslash(r, x_o, gauge);
      bistabcg_give_rr<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                         set_ptr->stream>>>(r, b__o, r_tilde, device_vals);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    }
    if (if_input == 0) {
      checkCudaErrors(cudaFreeAsync(b_e, set_ptr->stream));
      checkCudaErrors(cudaFreeAsync(b_o, set_ptr->stream));
    }
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
  void give(LatticeSet *_set_ptr) {
    set_ptr = _set_ptr;
    wilson_dslash.give(set_ptr);
  }
  void _wilson_dslash(void *fermion_out, void *fermion_in, void *gauge) {
    // src_o-_KAPPA_**2*dslash_oe(dslash_eo(src_o))
    wilson_dslash.run_eo(device_vec0, fermion_in, gauge);
    wilson_dslash.run_oe(device_vec1, device_vec0, gauge);
    bistabcg_give_dest_o<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                           set_ptr->stream>>>(
        fermion_out, fermion_in, device_vec1, _KAPPA_, device_vals);
  }
  void init(void *_x, void *_b, void *_gauge) {
    _init();
    if_input = 1;
    gauge = _gauge;
    x_e = _x;
    x_o = ((static_cast<LatticeComplex *>(_x)) + set_ptr->lat_4dim_SC);
    b_e = _b;
    b_o = ((static_cast<LatticeComplex *>(_b)) + set_ptr->lat_4dim_SC);
    __init();
  }
  void init(void *_gauge) {
    _init();
    if_input = 0;
    gauge = _gauge;
    __init();
  }
  void _dot(void *vec0, void *vec1, const int vals_index,
            const int stream_index) {
    // dest(val) = _dot(A,B)
    CUBLAS_CHECK(cublasDotcEx(
        set_ptr->cublasHs[stream_index], set_ptr->lat_4dim_SC, vec0,
        traits<data_type>::cuda_data_type, 1, vec1,
        traits<data_type>::cuda_data_type, 1,
        ((static_cast<LatticeComplex *>(device_vals)) + _send_tmp_),
        traits<data_type>::cuda_data_type, traits<data_type>::cuda_data_type));
    checkNcclErrors(ncclAllReduce(
        ((static_cast<LatticeComplex *>(device_vals)) + _send_tmp_),
        ((static_cast<LatticeComplex *>(device_vals)) + vals_index), 2,
        ncclDouble, ncclSum, set_ptr->nccl_comm,
        set_ptr->streams[stream_index]));
  }
  void _diff(void *x, void *ans) { // there is a bug
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
    _dot(ans, ans, _norm2_tmp_, _a_);
    bistabcg_give_diff<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                         set_ptr->streams[_a_]>>>(x, ans, device_vec0,
                                                  device_vals);
    _dot(device_vec0, device_vec0, _diff_tmp_, _a_);
    bistabcg_give_1diff<<<1, 1, 0, set_ptr->streams[_a_]>>>(device_vals);
    print_vals(999);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
  }
  void print_vals(int loop = 0) {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
    checkCudaErrors(
        cudaMemcpyAsync((static_cast<LatticeComplex *>(host_vals)),
                        (static_cast<LatticeComplex *>(device_vals)),
                        _vals_size_ * sizeof(LatticeComplex),
                        cudaMemcpyDeviceToHost, set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    std::cout << "######TIME :" << set_ptr->get_time() << "######" << std::endl
              << "##RANK     :" << set_ptr->node_rank << "##LOOP:" << loop
              << std::endl
              << "##tmp0     :" << host_vals[_tmp0_] << std::endl
              << "##tmp1     :" << host_vals[_tmp1_] << std::endl
              << "##rho_prev :" << host_vals[_rho_prev_] << std::endl
              << "##rho      :" << host_vals[_rho_] << std::endl
              << "##alpha    :" << host_vals[_alpha_] << std::endl
              << "##beta     :" << host_vals[_beta_] << std::endl
              << "##omega    :" << host_vals[_omega_] << std::endl
              << "##send_tmp :" << host_vals[_send_tmp_] << std::endl
              << "##norm2_tmp:" << host_vals[_norm2_tmp_] << std::endl
              << "##diff_tmp :" << host_vals[_diff_tmp_] << std::endl
              << "##lat_xyzt :" << host_vals[_lat_xyzt_] << std::endl;
    // exit(1);
  }
  void run_nccl() {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
    for (int loop = 0; loop < _MAX_ITER_; loop++) {
      _dot(r_tilde, r, _rho_, _a_);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
      {
        // beta = (rho / rho_prev) * (alpha / omega);
        bistabcg_give_1beta<<<1, 1, 0, set_ptr->streams[_a_]>>>(device_vals);
      }
      checkCudaErrors(cudaStreamSynchronize(
          set_ptr->streams[_a_])); // needed, but don't know why.
      {
        // rho_prev = rho;
        bistabcg_give_1rho_prev<<<1, 1, 0, set_ptr->streams[_b_]>>>(
            device_vals);
      }
      {
        // p[i] = r[i] + (p[i] - v[i] * omega) * beta;
        bistabcg_give_p<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                          set_ptr->streams[_a_]>>>(p, r, v, device_vals);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      _dot(r, r, _norm2_tmp_, _c_);
      {
        // v = A * p;
        _wilson_dslash(v, p, gauge);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      _dot(r_tilde, v, _tmp0_, _d_);
      {
        // alpha = rho / tmp0;
        bistabcg_give_1alpha<<<1, 1, 0, set_ptr->streams[_d_]>>>(device_vals);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
      {
        // s[i] = r[i] - v[i] * alpha;
        bistabcg_give_s<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                          set_ptr->streams[_a_]>>>(s, r, v, device_vals);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      {
        // t = A * s;
        _wilson_dslash(t, s, gauge);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      _dot(t, s, _tmp0_, _c_);
      _dot(t, t, _tmp1_, _d_);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
      {
        // break;
        checkCudaErrors(cudaMemcpyAsync(
            ((static_cast<LatticeComplex *>(host_vals)) + _norm2_tmp_),
            ((static_cast<LatticeComplex *>(device_vals)) + _norm2_tmp_),
            sizeof(LatticeComplex), cudaMemcpyDeviceToHost,
            set_ptr->streams[_d_]));
      }
      {
        // omega = tmp0 / tmp1;
        bistabcg_give_1omega<<<1, 1, 0, set_ptr->streams[_d_]>>>(device_vals);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
      {
        // r[i] = s[i] - t[i] * omega;
        bistabcg_give_r<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                          set_ptr->streams[_a_]>>>(r, s, t, device_vals);
      }
      {
        // x_o[i] = x_o[i] + p[i] * alpha + s[i] * omega;
        bistabcg_give_x_o<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                            set_ptr->streams[_b_]>>>(x_o, p, s, device_vals);
      }
      {
#ifdef PRINT_NCCL_WILSON_BISTABCG
        std::cout << "##RANK:" << set_ptr->node_rank << "##LOOP:" << loop
                  << "##Residual:" << host_vals[_norm2_tmp_]._data.x
                  << std::endl;
#endif
        if ((host_vals[_norm2_tmp_]._data.x < _TOL_ ||
             loop == _MAX_ITER_ - 1)) {
          std::cout << "##RANK:" << set_ptr->node_rank << "##LOOP:" << loop
                    << "##Residual:" << host_vals[_norm2_tmp_] << std::endl;
          break;
        }
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
      if (if_input) {
        // get $x_{e}$ by $b_{e}+\kappa D_{eo}x_{o}$
        CUBLAS_CHECK(cublasDcopy(set_ptr->cublasH,
                                 set_ptr->lat_4dim_SC * sizeof(data_type) /
                                     sizeof(double),
                                 (double *)b_e, 1, (double *)device_vec0, 1));
        wilson_dslash.run_eo(device_vec1, x_o, gauge);
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
        LatticeComplex _(_KAPPA_, 0.0);
        // dest(B) = B + alpha*A
        CUBLAS_CHECK(
            cublasAxpyEx(set_ptr->cublasH, set_ptr->lat_4dim_SC, &_,
                         traits<data_type>::cuda_data_type, device_vec1,
                         traits<data_type>::cuda_data_type, 1, device_vec0,
                         traits<data_type>::cuda_data_type, 1,
                         traits<data_type>::cuda_data_type));
        CUBLAS_CHECK(cublasDcopy(set_ptr->cublasH,
                                 set_ptr->lat_4dim_SC * sizeof(data_type) /
                                     sizeof(double),
                                 (double *)device_vec0, 1, (double *)x_e, 1));
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
      }
    }
  }
  void run_nccl_just_cg() {
    // D dag wait to do......
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
    // p[i] = r[i]
    CUBLAS_CHECK(
        cublasDcopy(set_ptr->cublasH,
                    set_ptr->lat_4dim_SC * sizeof(data_type) / sizeof(double),
                    (double *)r, 1, (double *)p, 1));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    for (int loop = 0; loop < _MAX_ITER_; loop++) {
      {
        // rho = <r, r>;
        _dot(r, r, _rho_, _a_);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
      {
        // v = A * p;
        _wilson_dslash(v, p, gauge);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      // tmp0 = <p ,Ap> = <p, v>;
      _dot(p, v, _tmp0_, _b_);
      {
        // alpha = <r, r>/<p ,Ap> = rho/tmp0;
        cg_give_1alpha<<<1, 1, 0, set_ptr->streams[_b_]>>>(device_vals);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
      {
        // x_o[i] = x_o[i] + v * alpha;
        cg_give_x_o<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                      set_ptr->streams[_c_]>>>(x_o, p, device_vals);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
      {
        // r_tilde[i] = r[i] - v * alpha;
        // r[i] = r_tilde[i]
        cg_give_rr<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                     set_ptr->streams[_d_]>>>(r, r_tilde, v, device_vals);
      }
      {
        // rho_prev = <r_tilde, r_tilde>;
        _dot(r_tilde, r_tilde, _rho_prev_, _d_);
      }
      {
        // break;
        checkCudaErrors(cudaMemcpyAsync(
            ((static_cast<LatticeComplex *>(host_vals)) + _rho_prev_),
            ((static_cast<LatticeComplex *>(device_vals)) + _rho_prev_),
            sizeof(LatticeComplex), cudaMemcpyDeviceToHost,
            set_ptr->streams[_d_]));
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
      {
        // beta = <r_tilde, r_tilde>/<r, r> = rho_prev/rho;
        cg_give_1beta<<<1, 1, 0, set_ptr->streams[_b_]>>>(device_vals);
      }
      {
        // p[i] = r_tilde[i] + p[i] * beta
        cg_give_p<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                    set_ptr->streams[_b_]>>>(p, r_tilde, device_vals);
      }
      {
#ifdef PRINT_NCCL_WILSON_BISTABCG
        std::cout << "##RANK:" << set_ptr->node_rank << "##LOOP:" << loop
                  << "##Residual:" << host_vals[_rho_prev_]._data.x
                  << std::endl;
#endif
      }
      if ((host_vals[_rho_prev_]._data.x < _TOL_ || loop == _MAX_ITER_ - 1)) {
        std::cout << "##RANK:" << set_ptr->node_rank << "##LOOP:" << loop
                  << "##Residual:" << host_vals[_rho_prev_] << std::endl;
        break;
      }
    }
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
    if (if_input) {
      // get $x_{e}$ by $b_{e}+\kappa D_{eo}x_{o}$
      CUBLAS_CHECK(
          cublasDcopy(set_ptr->cublasH,
                      set_ptr->lat_4dim_SC * sizeof(data_type) / sizeof(double),
                      (double *)b_e, 1, (double *)device_vec0, 1));
      wilson_dslash.run_eo(device_vec1, x_o, gauge);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      LatticeComplex _(_KAPPA_, 0.0);
      // dest(B) = B + alpha*A
      CUBLAS_CHECK(cublasAxpyEx(set_ptr->cublasH, set_ptr->lat_4dim_SC, &_,
                                traits<data_type>::cuda_data_type, device_vec1,
                                traits<data_type>::cuda_data_type, 1,
                                device_vec0, traits<data_type>::cuda_data_type,
                                1, traits<data_type>::cuda_data_type));
      CUBLAS_CHECK(
          cublasDcopy(set_ptr->cublasH,
                      set_ptr->lat_4dim_SC * sizeof(data_type) / sizeof(double),
                      (double *)device_vec0, 1, (double *)x_e, 1));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
    }
  }
  void _run() {
    auto start = std::chrono::high_resolution_clock::now();
    run_nccl();
    // run_nccl_just_cg();
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
  }
  void run() {
#ifdef PRINT_NCCL_WILSON_BISTABCG
    set_ptr->_print();
#endif
    _run();
    if (if_input == 0) {
      _diff(x_o, ans_o);
    } else {
      _wilson_dslash(device_vec1, x_o, gauge);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      _diff(device_vec1, b__o);
    }
  }
  void end() {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
    if (if_input == 0) {
      checkCudaErrors(cudaFreeAsync(ans_e, set_ptr->stream));
      checkCudaErrors(cudaFreeAsync(ans_o, set_ptr->stream));
      checkCudaErrors(cudaFreeAsync(x_o, set_ptr->stream));
    }
    checkCudaErrors(cudaFreeAsync(b__o, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(r, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(r_tilde, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(p, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(v, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(s, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(t, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(device_vec0, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(device_vec1, set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(device_vals, set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
  }
};
#endif
