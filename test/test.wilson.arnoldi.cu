#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
using T = float;
int main()
{
  MPI_Init(NULL, NULL);
  void *gauge;
  QcuParam param;
  QcuParam grid;
  int parity, lat_4dim_DCC;
  { // io
    std::stringstream filename;
    filename << "wilson-bistabcg-gauge_-32-32-32-32-1048576-1-1-1-1-0-0-1-0-f.bin";
    get_filename(filename, param, parity, grid);
    lat_4dim_DCC = param.lattice_size[_X_] * param.lattice_size[_Y_] * param.lattice_size[_Z_] * param.lattice_size[_T_] * _LAT_DCC_ / _EVEN_ODD_;
    { // gauge
      std::stringstream filename;
      filename << "wilson-bistabcg-gauge_-32-32-32-32-1048576-1-1-1-1-0-0-1-0-f.bin";
      cudaDeviceSynchronize();
      cudaMalloc(&gauge, lat_4dim_DCC * _EVEN_ODD_ * _REAL_IMAG_ * sizeof(T));
      cudaDeviceSynchronize();
      device_load<T>(gauge, lat_4dim_DCC * _EVEN_ODD_ * _REAL_IMAG_, filename.str());
    }
  }
  { // test
    LatticeSet<T> _set;
    _set.give(param.lattice_size, grid.lattice_size);
    _set.init();
    dptzyxcc2ccdptzyx<T>(gauge, &_set);
    LatticeBistabCg<T> _bistabcg;
    _bistabcg.give(&_set);
    _bistabcg.init(gauge);
    {
      _bistabcg.run_test();
      {
        LatticeArnoldi<T> _arnoldi;
        _arnoldi.give(&_bistabcg);
        _arnoldi.run_test();
      }
    }
    _bistabcg.end();
    ccdptzyx2dptzyxcc<T>(gauge, &_set);
    _set.end();
  }
  cudaFree(gauge);

  MPI_Finalize();
  return 0;
}