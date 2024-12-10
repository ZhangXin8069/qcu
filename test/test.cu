#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
using T = double;
int main()
{
  MPI_Init(NULL, NULL);
  void *gauge = nullptr;
  void *fermion_in = nullptr;
  void *fermion_out = nullptr;
  QcuParam param;
  QcuParam grid;
  int parity;
  { // io
    std::stringstream filename;
    filename << "wilson-clover-dslash-kappa1-gauge_1733741679_-16-16-32-32-262144-1-1-1-1-1-0-1-0-d.bin";
    get_filename(filename, param, parity, grid);
  }
  // define for apply_clover_dslash
  LatticeSet<T> _set;
  _set.give(param.lattice_size, grid.lattice_size, parity);
  _set.init();
  _set._print();
  { // io
    std::stringstream filename;
    filename << "wilson-clover-dslash-kappa1-fermion-out_1733741679_-16-16-32-32-262144-1-1-1-1-1-0-1-0-d.bin";
    device_load<T>(fermion_out, _set.lat_4dim_SC * _REAL_IMAG_, filename.str());
  }
  { // io
    std::stringstream filename;
    filename << "wilson-clover-dslash-kappa1-fermion-in_1733741679_-16-16-32-32-262144-1-1-1-1-1-0-1-0-d.bin";
    device_load<T>(fermion_in, _set.lat_4dim_SC * _REAL_IMAG_, filename.str());
  }
  { // io
    std::stringstream filename;
    filename << "wilson-clover-dslash-kappa1-gauge_1733741679_-16-16-32-32-262144-1-1-1-1-1-0-1-0-d.bin";
    device_load<T>(gauge, _set.lat_4dim_DCC * _EVEN_ODD_ * _REAL_IMAG_, filename.str());
  }
  LatticeWilsonDslash<T> _wilson_dslash;
  LatticeCloverDslash<T> _clover_dslash;
  _wilson_dslash.give(&_set);
  _clover_dslash.give(&_set);
  _clover_dslash.init();
  // {
  //   // wilson dslash
  //   _wilson_dslash.run_test(fermion_out, fermion_in, gauge);
  // }
  // {
  //   // make clover
  //   _clover_dslash.make(gauge);
  // }
  // {
  //   // inverse clover
  //   _clover_dslash.inverse();
  // }
  // {
  //   // give clover
  //   _clover_dslash.give(fermion_out);
  // }
  { // io
    std::stringstream filename;
    filename << "_wilson-clover-dslash-kappa1-fermion-out_1733741679_-16-16-32-32-262144-1-1-1-1-1-0-1-0-d.bin";
    device_save<T>(fermion_out, _set.lat_4dim_SC * _REAL_IMAG_, filename.str());
  }
  _clover_dslash.end();
  _set.end();
  cudaFree(gauge);
  cudaFree(fermion_in);
  cudaFree(fermion_out);
  MPI_Finalize();
  return 0;
}