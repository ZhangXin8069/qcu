#ifndef _LATTICE_ARNOLDI_H
#define _LATTICE_ARNOLDI_H
#include "./arnoldi.h"
#include "./lattice_bistabcg.h"
namespace qcu
{
  template <typename T>
  struct LatticeArnoldi
  {
    void *Q, *H, *gauge;
    LatticeSet<T> *set_ptr;
    LatticeWilsonDslash<T> dslash;
    void give(LatticeSet<T> *_set_ptr)
    {
      set_ptr = _set_ptr;
      dslash.give(set_ptr);
    }
    void init(void *_gauge)
    {
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      gauge = _gauge;
      checkCudaErrors(
          cudaMallocAsync(&Q, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
                          set_ptr->stream));
      H = (void *)malloc(
          );
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    }
    void _run()
    {
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      dslash._run_init(); // give init with new r (b__o)
      dslash._run();      // give e (x_o) [A * e = r]
      gmres_ir_give_x<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                           set_ptr->stream>>>(x, e, dslash.device_vals); // x = x + e
      dslash._wilson_dslash(r, x, dslash.gauge);                         // r (tmp use) = A * x
      gmres_ir_give_r<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                           set_ptr->stream>>>(r, b, dslash.device_vals); // r = b -r (b - A *x)
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
        dslash._dot(r, r, _norm2_tmp_, _a_);
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
        std::cout << "##RANK:" << set_ptr->host_params[_NODE_RANK_] << "##LOOP:" << loop
                  << "##Residual:" << dslash.host_vals[_norm2_tmp_] << std::endl;
        if ((dslash.host_vals[_norm2_tmp_]._data.x < set_ptr->tol() / 10 ||
             loop == set_ptr->max_iter() - 1)) // just for test, wait for multi-precision
        {
          std::cout << "##RANK:" << set_ptr->host_params[_NODE_RANK_] << "##LOOP:" << loop
                    << "##Residual:" << dslash.host_vals[_norm2_tmp_] << std::endl;
          break;
        }
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,
                                  set_ptr->lat_4dim_SC * _REAL_IMAG_,
                                  (T *)x, 1, (T *)dslash.x_o, 1)); // x -> x_o
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      dslash.run(); // give x_e
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
      dslash._wilson_dslash(dslash.device_vec1, dslash.x_o, dslash.gauge);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      dslash._diff(dslash.device_vec1, dslash.b__o);
    }
    void end()
    {
      dslash.end();
    }
  };
}
#endif
