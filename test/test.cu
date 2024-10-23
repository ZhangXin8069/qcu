#include "./include/qcu.h"
#include "define.h"
// #define __CLOVER_DSLASH__
int main()
{
  MPI_Init(NULL, NULL);
  int param_lattice_size[_QCU_DIM_];
  int grid_lattice_size[_QCU_DIM_];
  for (int i = 0; i < _QCU_DIM_; i++)
  {
    param_lattice_size[i] = _QCU_LAT_EXAMPLE_;
    grid_lattice_size[i] = _QCU_GRID_EXAMPLE_;
  }
  // grid_lattice_size[_T_] = 2;
  LatticeSet _set;
  int parity = 1;
  void *gauge;
  void *fermion_in;
  void *fermion_out;
  _set.give(param_lattice_size, grid_lattice_size, parity);
  _set.init();
  checkCudaErrors(cudaMalloc(
      &gauge, _QCU_LAT_DCC_ * _QCU_EVEN_ODD_ * _QCU_LAT_EXAMPLE_ * _QCU_LAT_EXAMPLE_ *
                  _QCU_LAT_EXAMPLE_ * _QCU_LAT_EXAMPLE_ * sizeof(LatticeComplex)));
  checkCudaErrors(cudaStreamSynchronize(_set.stream));
  give_debug_u<<<_set.gridDim, _set.blockDim, 0, _set.stream>>>(
      gauge, _set.device_params);
  checkCudaErrors(cudaStreamSynchronize(_set.stream));
  checkCudaErrors(cudaMalloc(
      &fermion_in, _QCU_LAT_SC_ * _QCU_EVEN_ODD_ * _QCU_LAT_EXAMPLE_ * _QCU_LAT_EXAMPLE_ *
                       _QCU_LAT_EXAMPLE_ * _QCU_LAT_EXAMPLE_ * sizeof(LatticeComplex)));
  checkCudaErrors(cudaMalloc(&fermion_out, _QCU_LAT_SC_ * _QCU_EVEN_ODD_ *
                                               _QCU_LAT_EXAMPLE_ * _QCU_LAT_EXAMPLE_ *
                                               _QCU_LAT_EXAMPLE_ * _QCU_LAT_EXAMPLE_ *
                                               sizeof(LatticeComplex)));
  {
    // define for dslash
    dptzyxcc2ccdptzyx(gauge, &_set);
    tzyxsc2sctzyx(fermion_in, &_set);
    tzyxsc2sctzyx(fermion_out, &_set);
    LatticeWilsonDslash _wilson_dslash;
    _wilson_dslash.give(&_set);
#ifdef __CLOVER_DSLASH__
    LatticeCloverDslash _clover_dslash;
    _clover_dslash.give(&_set);
    _clover_dslash.init();
#endif
    {
      // // wilson dslash
      // _wilson_dslash.run_test(fermion_out, fermion_in, gauge);
      LatticeBistabcg _bistabcg;
      _bistabcg.give(&_set);
      _bistabcg.init(fermion_out, fermion_in, gauge);
      _bistabcg.run();
      _bistabcg.end();
    }
#ifdef __CLOVER_DSLASH__
    {
      // make clover
      _clover_dslash.make(gauge);
    }
    {
      // inverse clover
      _clover_dslash.inverse();
    }
    {
      // give clover
      _clover_dslash.give(fermion_out);
    }
#endif
    ccdptzyx2dptzyxcc(gauge, &_set);
    sctzyx2tzyxsc(fermion_in, &_set);
    sctzyx2tzyxsc(fermion_out, &_set);
#ifdef __CLOVER_DSLASH__
    _clover_dslash.end();
#endif
  }
  cudaFree(gauge);
  cudaFree(fermion_in);
  cudaFree(fermion_out);
  _set.end();
  MPI_Finalize();
  return 0;
}