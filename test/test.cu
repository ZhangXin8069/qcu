#include "./include/qcu.h"
#include "define.h"
// #define __CLOVER_DSLASH__
int main() {
  MPI_Init(NULL, NULL);
  int param_lattice_size[_DIM_];
  int grid_lattice_size[_DIM_];
  for (int i = 0; i < _DIM_; i++) {
    param_lattice_size[i] = _LAT_EXAMPLE_;
    grid_lattice_size[i] = _GRID_EXAMPLE_;
  }
  // grid_lattice_size[_T_] = 2;
  LatticeSet _set;
  _set.give(param_lattice_size, grid_lattice_size);
  _set.init();
  int parity = 1;
  void *gauge;
  void *fermion_in;
  void *fermion_out;
  checkCudaErrors(cudaMalloc(
      &gauge, _LAT_DCC_ * _EVEN_ODD_ * _LAT_EXAMPLE_ * _LAT_EXAMPLE_ *
                  _LAT_EXAMPLE_ * _LAT_EXAMPLE_ * sizeof(LatticeComplex)));
  checkCudaErrors(cudaStreamSynchronize(_set.stream));
  give_debug_u<<<_set.gridDim, _set.blockDim, 0, _set.stream>>>(
      gauge, _set.device_lat_xyzt, parity, _set.node_rank);
  checkCudaErrors(cudaStreamSynchronize(_set.stream));
  checkCudaErrors(cudaMalloc(
      &fermion_in, _LAT_SC_ * _EVEN_ODD_ * _LAT_EXAMPLE_ * _LAT_EXAMPLE_ *
                       _LAT_EXAMPLE_ * _LAT_EXAMPLE_ * sizeof(LatticeComplex)));
  checkCudaErrors(cudaMalloc(&fermion_out, _LAT_SC_ * _EVEN_ODD_ *
                                               _LAT_EXAMPLE_ * _LAT_EXAMPLE_ *
                                               _LAT_EXAMPLE_ * _LAT_EXAMPLE_ *
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
      // wilson dslash
      _wilson_dslash.run_test(fermion_out, fermion_in, gauge, parity);
    }
#ifdef __CLOVER_DSLASH__
    {
      // make clover
      _clover_dslash.make(gauge, parity);
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