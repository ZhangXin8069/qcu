#ifndef _LATTICE_GMERS_IR_H
#define _LATTICE_GMERS_IR_H
#include "./gmres_ir.h"
#include "./lattice_bistabcg.h"
namespace qcu
{
  template <typename T>
  struct LatticeGmresIr
  {
    void *r, *b, *x, *e;
    LatticeSet<T> *set_ptr;
    LatticeBistabCg<T> bistabcg;
    void give(LatticeSet<T> *_set_ptr)
    {
      set_ptr = _set_ptr;
      bistabcg.give(set_ptr);
    }
    void init(void *_x, void *_b, void *_gauge)
    {
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      bistabcg.init(_x, _b, _gauge); // r_tilde = r = b__o (real b) - A * x_o, r_tilde and b__o doesn't change after init.
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      checkCudaErrors(
          cudaMallocAsync(&x, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
                          set_ptr->stream));
      CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,
                                  set_ptr->lat_4dim_SC * _REAL_IMAG_,
                                  (T *)bistabcg.x_o, 1, (T *)x, 1)); // x_o (x_0) -> x
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      checkCudaErrors(
          cudaMallocAsync(&b, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
                          set_ptr->stream));
      CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,
                                  set_ptr->lat_4dim_SC * _REAL_IMAG_,
                                  (T *)bistabcg.b__o, 1, (T *)b, 1)); // b__o (real b) -> b
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      r = bistabcg.b__o; // r <-> b__o
      CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,
                                  set_ptr->lat_4dim_SC * _REAL_IMAG_,
                                  (T *)bistabcg.r_tilde, 1, (T *)r, 1)); // r_tilde (b__o - A * x_o) [b - A * x_0] -> r (r_0)
      e = bistabcg.x_o;                                                  // e <-> x_o
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    }
    void _run()
    {
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      bistabcg._run_init(); // give init with new r (b__o)
      bistabcg._run();      // give e (x_o) [A * e = r]
      gmres_ir_give_x<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                           set_ptr->stream>>>(x, e, bistabcg.device_vals); // x = x + e
      bistabcg._wilson_dslash(r, x, bistabcg.gauge);                       // r (tmp use) = A * x
      gmres_ir_give_r<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                           set_ptr->stream>>>(r, b, bistabcg.device_vals); // r = b -r (b - A *x)
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    }
    void run()
    {
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      for (int loop = 0; loop < set_ptr->max_iter(); loop++)
      {
        _run();
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
        bistabcg._dot(r, r, _norm2_tmp_, _a_);
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
        std::cout << "##RANK:" << set_ptr->host_params[_NODE_RANK_] << "##LOOP:" << loop
                  << "##Residual:" << bistabcg.host_vals[_norm2_tmp_] << std::endl;
        if ((bistabcg.host_vals[_norm2_tmp_]._data.x < set_ptr->tol() / 10 ||
             loop == set_ptr->max_iter() - 1)) // just for test, wait for multi-precision
        {
          std::cout << "##RANK:" << set_ptr->host_params[_NODE_RANK_] << "##LOOP:" << loop
                    << "##Residual:" << bistabcg.host_vals[_norm2_tmp_] << std::endl;
          break;
        }
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,
                                  set_ptr->lat_4dim_SC * _REAL_IMAG_,
                                  (T *)x, 1, (T *)bistabcg.x_o, 1)); // x -> x_o
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      bistabcg.run(); // give x_e
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    }
    void run_test()
    {
      auto start = std::chrono::high_resolution_clock::now();
      run();
      auto end = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
      set_ptr->err = cudaGetLastError();
      checkCudaErrors(set_ptr->err);
      printf(
          "multi-gpu wilson gmres_ir total time: (without malloc free memcpy) :%.9lf "
          "sec\n",
          T(duration) / 1e9);
      bistabcg._wilson_dslash(bistabcg.device_vec1, bistabcg.x_o, bistabcg.gauge);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      bistabcg._diff(bistabcg.device_vec1, bistabcg.b__o);
    }
    void end()
    {
      bistabcg.end();
    }
  };
}
#endif
