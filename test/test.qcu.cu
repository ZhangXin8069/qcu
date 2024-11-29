#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
using T = double;
int main()
{
  MPI_Init(NULL, NULL);
  void *gauge;
  void *fermion_in;
  void *fermion_out;
  QcuParam param;
  QcuParam grid;
  for (int i = 0; i < _DIM_; i++)
  {
    param.lattice_size[i] = _LAT_EXAMPLE_;
    grid.lattice_size[i] = _GRID_EXAMPLE_;
  }
  grid.lattice_size[_T_] = 2;
  checkCudaErrors(cudaMalloc(
      &gauge, _LAT_DCC_ * _EVEN_ODD_ * _LAT_EXAMPLE_ * _LAT_EXAMPLE_ *
                  _LAT_EXAMPLE_ * _LAT_EXAMPLE_ * sizeof(LatticeComplex<T>)));
  LatticeSet<T> _set;
  int parity = _ODD_;
  _set.give(param.lattice_size, grid.lattice_size, parity);
  _set.init();
  give_debug_u<T><<<_set.gridDim, _set.blockDim, 0, _set.stream>>>(
      gauge, _set.device_params);
  checkCudaErrors(cudaStreamSynchronize(_set.stream));
  checkCudaErrors(cudaMalloc(
      &fermion_in, _LAT_SC_ * _EVEN_ODD_ * _LAT_EXAMPLE_ * _LAT_EXAMPLE_ *
                       _LAT_EXAMPLE_ * _LAT_EXAMPLE_ * sizeof(LatticeComplex<T>)));
  checkCudaErrors(cudaMalloc(&fermion_out, _LAT_SC_ * _EVEN_ODD_ *
                                               _LAT_EXAMPLE_ * _LAT_EXAMPLE_ *
                                               _LAT_EXAMPLE_ * _LAT_EXAMPLE_ *
                                               sizeof(LatticeComplex<T>)));
  applyCloverDslashQcu(fermion_out, fermion_in, gauge,
                       &param, parity, &grid);
  cudaFree(gauge);
  cudaFree(fermion_in);
  cudaFree(fermion_out);
  _set.end();
  MPI_Finalize();
  return 0;
}