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
  int parity;
  { // io
    std::stringstream filename;
    filename << "wilson-bistabcg-mass0-gauge_1733741679_-16-32-32-32-524288-1-1-1-1-16777216-0-1-0-d.bin";
    get_filename(filename, param, parity, grid);
  }
  LatticeSet<T> _set;
  _set.give(param.lattice_size, grid.lattice_size, parity);
  _set.init();
  _set._print();
  {   // test
    { // io
      std::stringstream filename;
      filename << "wilson-clover-dslash-kappa1-fermion-out";
      give_filename(filename, _set.host_params);
      device_save<T>(fermion_out, _set.lat_4dim_SC * _REAL_IMAG_, filename.str());
    }
    { // io
      std::stringstream filename;
      filename << "wilson-clover-dslash-kappa1-fermion-in";
      give_filename(filename, _set.host_params);
      device_save<T>(fermion_in, _set.lat_4dim_SC * _REAL_IMAG_, filename.str());
    }
    { // io
      std::stringstream filename;
      filename << "wilson-clover-dslash-kappa1-gauge";
      give_filename(filename, _set.host_params);
      device_save<T>(gauge, _set.lat_4dim_SC * _REAL_IMAG_ * _EVEN_ODD_, filename.str());
    }
  }
  _set.end();
  cudaFree(gauge);
  cudaFree(fermion_in);
  cudaFree(fermion_out);
  MPI_Finalize();
  return 0;
}