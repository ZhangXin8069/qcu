#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
using T = double;
void applyCloverDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                          QcuParam *param, int parity, QcuParam *grid)
{
  // define for apply_clover_dslash
  LatticeSet<T> _set;
  _set.give(param->lattice_size, grid->lattice_size, parity);
  _set.init();
  dptzyxcc2ccdptzyx<T>(gauge, &_set);
  tzyxsc2sctzyx<T>(fermion_in, &_set);
  tzyxsc2sctzyx<T>(fermion_out, &_set);
  LatticeWilsonDslash<T> _wilson_dslash;
  LatticeCloverDslash<T> _clover_dslash;
  _wilson_dslash.give(&_set);
  _clover_dslash.give(&_set);
  _clover_dslash.init();
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
    exit(1);
  }
  {
    // wilson dslash
    _wilson_dslash.run_test(fermion_out, fermion_in, gauge);
  }
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
  ccdptzyx2dptzyxcc<T>(gauge, &_set);
  sctzyx2tzyxsc<T>(fermion_in, &_set);
  sctzyx2tzyxsc<T>(fermion_out, &_set);
  _clover_dslash.end();
  _set.end();
}