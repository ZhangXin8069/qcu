#ifndef _LATTICE_BISTABCG_H
#define _LATTICE_BISTABCG_H
#include "./bistabcg.h"
#include "./dslash.h"
#include "./lattice_cuda.h"
#include "./lattice_dslash.h"
#include "./lattice_mpi.h"

struct LatticeBistabcg {
  LatticeSet *set_ptr;
  LatticeWilsonDslash dslash;
  LatticeComplex r_norm2;
  LatticeComplex rho_prev;
  LatticeComplex rho;
  LatticeComplex alpha;
  LatticeComplex omega;
  LatticeComplex beta;
  LatticeComplex tmp;
  LatticeComplex tmp0;
  LatticeComplex tmp1;
  LatticeComplex local_result;
  LatticeComplex *host_tmps0;
  LatticeComplex *host_tmps1;
  LatticeComplex *host_dots;
  void *ans_e, *ans_o, *x_e, *x_o, *b_e, *b_o, *b__o, *r, *r_tilde, *p, *v, *s,
      *t, *device_tmps0, *device_tmps1, *device_dots;
  void _init() {
    r_norm2.real = 0.0;
    r_norm2.imag = 0.0;
    rho_prev.real = 1.0;
    rho_prev.imag = 0.0;
    rho.real = 0.0;
    rho.imag = 0.0;
    alpha.real = 1.0;
    alpha.imag = 0.0;
    omega.real = 1.0;
    omega.imag = 0.0;
    beta.real = 0.0;
    beta.imag = 0.0;
    tmp.real = 0.0;
    tmp.imag = 0.0;
    tmp0.real = 0.0;
    tmp0.imag = 0.0;
    tmp1.real = 0.0;
    tmp1.imag = 0.0;
    local_result.real = 0.0;
    local_result.imag = 0.0;
    cudaMalloc(&ans_e, set_ptr->lat_4dim12 * sizeof(LatticeComplex));
    cudaMalloc(&ans_o, set_ptr->lat_4dim12 * sizeof(LatticeComplex));
    cudaMalloc(&x_e, set_ptr->lat_4dim12 * sizeof(LatticeComplex));
    cudaMalloc(&x_o, set_ptr->lat_4dim12 * sizeof(LatticeComplex));
    cudaMalloc(&b__o, set_ptr->lat_4dim12 * sizeof(LatticeComplex));
    cudaMalloc(&r, set_ptr->lat_4dim12 * sizeof(LatticeComplex));
    cudaMalloc(&r_tilde, set_ptr->lat_4dim12 * sizeof(LatticeComplex));
    cudaMalloc(&p, set_ptr->lat_4dim12 * sizeof(LatticeComplex));
    cudaMalloc(&v, set_ptr->lat_4dim12 * sizeof(LatticeComplex));
    cudaMalloc(&s, set_ptr->lat_4dim12 * sizeof(LatticeComplex));
    cudaMalloc(&t, set_ptr->lat_4dim12 * sizeof(LatticeComplex));
    cudaMalloc(&device_tmps0, set_ptr->lat_4dim12 * sizeof(LatticeComplex));
    cudaMalloc(&device_tmps1, set_ptr->lat_4dim12 * sizeof(LatticeComplex));
    cudaMalloc(&device_dots, set_ptr->lat_4dim * sizeof(LatticeComplex));
    host_tmps0 =
        (LatticeComplex *)malloc(set_ptr->lat_4dim12 * sizeof(LatticeComplex));
    host_tmps1 =
        (LatticeComplex *)malloc(set_ptr->lat_4dim12 * sizeof(LatticeComplex));
    host_dots =
        (LatticeComplex *)malloc(set_ptr->lat_4dim * sizeof(LatticeComplex));
    give_random_value<<<set_ptr->gridDim, set_ptr->blockDim>>>(x_o, 1314999);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  void give(LatticeSet *_set_ptr) {
    set_ptr = _set_ptr;
    dslash.give(set_ptr);
    _init();
    // set_ptr->_print();
  }
  void _dslash(void *fermion_out, void *fermion_in, void *gauge) {
    // src_o-_KAPPA_**2*dslash_oe(dslash_eo(src_o))
    dslash.run_eo(device_tmps0, fermion_in, gauge);
    dslash.run_oe(device_tmps1, device_tmps0, gauge);
    bistabcg_give_dest_o<<<set_ptr->gridDim, set_ptr->blockDim>>>(
        fermion_out, fermion_in, device_tmps1, _KAPPA_);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  void init(void *_b_e, void *_b_o, void *gauge) {
    b_e = _b_e;
    b_o = _b_o;
    dslash.run_oe(device_tmps0, b_e, gauge);
    bistabcg_give_b__0<<<set_ptr->gridDim, set_ptr->blockDim>>>(
        b__o, b_o, device_tmps0, _KAPPA_);
    checkCudaErrors(cudaDeviceSynchronize());
    _dslash(r, x_o, gauge);
    bistabcg_give_rr<<<set_ptr->gridDim, set_ptr->blockDim>>>(r, b__o, r_tilde);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  void init(void *gauge) {
    cudaMalloc(&b_e, set_ptr->lat_4dim12 * sizeof(LatticeComplex));
    cudaMalloc(&b_o, set_ptr->lat_4dim12 * sizeof(LatticeComplex));
    give_random_value<<<set_ptr->gridDim, set_ptr->blockDim>>>(ans_e, 8848);
    give_random_value<<<set_ptr->gridDim, set_ptr->blockDim>>>(ans_o, 12138);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamSynchronize(set_ptr->qcu_stream));
    dslash.run_eo(device_tmps0, ans_o, gauge);
    bistabcg_give_b_e<<<set_ptr->gridDim, set_ptr->blockDim>>>(
        b_e, ans_e, device_tmps0, _KAPPA_);
    dslash.run_oe(device_tmps1, ans_e, gauge);
    bistabcg_give_b_o<<<set_ptr->gridDim, set_ptr->blockDim>>>(
        b_o, ans_o, device_tmps1, _KAPPA_);
    dslash.run_oe(device_tmps0, b_e, gauge);
    bistabcg_give_b__0<<<set_ptr->gridDim, set_ptr->blockDim>>>(
        b__o, b_o, device_tmps0, _KAPPA_);
    checkCudaErrors(cudaDeviceSynchronize());
    _dslash(r, x_o, gauge);
    bistabcg_give_rr<<<set_ptr->gridDim, set_ptr->blockDim>>>(r, b__o, r_tilde);
    give_custom_value<<<set_ptr->gridDim, set_ptr->blockDim>>>(ans_o, 1.0,
                                                               0.0); // test b=1
    checkCudaErrors(cudaDeviceSynchronize());
  }
  void dot(void *val0, void *val1, LatticeComplex *dest_ptr) {
    LatticeComplex _(0.0, 0.0);
    bistabcg_part_dot<<<set_ptr->gridDim, set_ptr->blockDim>>>(device_dots,
                                                               val0, val1);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamSynchronize(set_ptr->qcu_stream));
    cudaMemcpy(host_dots, device_dots,
               sizeof(LatticeComplex) * set_ptr->lat_4dim,
               cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamSynchronize(set_ptr->qcu_stream));
    for (int i = 0; i < set_ptr->lat_4dim; i++) {
      _ += host_dots[i];
    }
    MPI_Allreduce(&_, dest_ptr, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  void print_norm2(void *val) {
    LatticeComplex _(0.0, 0.0);
    dot(val, val, &_);
    printf("#*%p*:%.9lf - from rank %d#\n", val, _.real, set_ptr->node_rank);
  }
  void diff(void *val0, void *val1, LatticeComplex *dest_ptr) {
    dest_ptr->real = 0.0;
    dest_ptr->imag = 0.0;
    tmp0.real = 0.0;
    tmp0.imag = 0.0;
    tmp1.real = 0.0;
    tmp1.imag = 0.0;
    local_result.real = 0.0;
    local_result.imag = 0.0;
    bistabcg_part_cut<<<set_ptr->gridDim, set_ptr->blockDim>>>(device_tmps0,
                                                               val0, val1);
    checkCudaErrors(cudaDeviceSynchronize());
    dot(device_tmps0, device_tmps0, &tmp0);
    dot(val1, val1, &tmp1);
    *dest_ptr = tmp0 / tmp1;
  }
  void run(void *gauge) {
    for (int loop = 0; loop < _MAX_ITER_; loop++) {
      dot(r_tilde, r, &rho);
#ifdef DEBUG_NCCL_WILSON_BISTABCG
      std::cout << "##RANK:" << set_ptr->node_rank << "##LOOP:" << loop
                << "##rho:" << rho.real << std::endl;
#endif
      beta = (rho / rho_prev) * (alpha / omega);
#ifdef DEBUG_NCCL_WILSON_BISTABCG
      std::cout << "##RANK:" << set_ptr->node_rank << "##LOOP:" << loop
                << "##beta:" << beta.real << std::endl;
#endif
      bistabcg_give_p<<<set_ptr->gridDim, set_ptr->blockDim>>>(p, r, v, omega,
                                                               beta);
      checkCudaErrors(cudaDeviceSynchronize());
      // v = A * p;
      _dslash(v, p, gauge);
      dot(r_tilde, v, &tmp);
      alpha = rho / tmp;
#ifdef DEBUG_NCCL_WILSON_BISTABCG
      std::cout << "##RANK:" << set_ptr->node_rank << "##LOOP:" << loop
                << "##alpha:" << alpha.real << std::endl;
#endif
      bistabcg_give_s<<<set_ptr->gridDim, set_ptr->blockDim>>>(s, r, v, alpha);
      checkCudaErrors(cudaDeviceSynchronize());
      // t = A * s;
      _dslash(t, s, gauge);
      dot(t, s, &tmp0);
      dot(t, t, &tmp1);
      omega = tmp0 / tmp1;
#ifdef DEBUG_NCCL_WILSON_BISTABCG
      std::cout << "##RANK:" << set_ptr->node_rank << "##LOOP:" << loop
                << "##omega:" << omega.real << std::endl;
#endif
      bistabcg_give_x_o<<<set_ptr->gridDim, set_ptr->blockDim>>>(x_o, p, s,
                                                                 alpha, omega);
      bistabcg_give_r<<<set_ptr->gridDim, set_ptr->blockDim>>>(r, s, t, omega);
      checkCudaErrors(cudaDeviceSynchronize());
      dot(r, r, &r_norm2);
#ifdef PRINT_NCCL_WILSON_BISTABCG
      std::cout << "##RANK:" << set_ptr->node_rank << "##LOOP:" << loop
                << "##Residual:" << r_norm2.real << std::endl;
#endif
      // break;
      if (r_norm2.real < _TOL_ || loop == _MAX_ITER_ - 1) {
        break;
      }
      rho_prev = rho;
    }
    checkCudaErrors(cudaDeviceSynchronize());
  }
  void run_test(void *gauge) {
#ifdef PRINT_NCCL_WILSON_BISTABCG
    set_ptr->_print();
#endif
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
    diff(x_o, ans_o, &tmp);
    printf("## difference: %.16f\n", tmp.real);
  }
  void end() {
    cudaFree(ans_e);
    cudaFree(ans_o);
    cudaFree(x_o);
    cudaFree(b__o);
    cudaFree(r);
    cudaFree(r_tilde);
    cudaFree(p);
    cudaFree(v);
    cudaFree(s);
    cudaFree(t);
    cudaFree(device_tmps0);
    cudaFree(device_tmps1);
    free(host_tmps0);
    free(host_tmps1);
  }
};

#endif