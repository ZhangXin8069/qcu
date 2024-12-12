// #include "../include/qcu.h"
// #pragma optimize(5)
// using namespace qcu;
// using T = float;
// int main()
// {
//   MPI_Init(NULL, NULL);
//   void *gauge;
//   void *fermion_in_even;
//   void *fermion_out_even;
//   void *fermion_in_odd;
//   void *fermion_out_odd;
//   QcuParam param;
//   QcuParam grid;
//   int parity;
//   { // io
//     std::stringstream filename;
//     filename << "wilson-clover-dslash-kappa1-gauge_1733904246_-16-16-32-32-524288-1-1-1-1-0-0-1-0-f.bin_";
//     get_filename(filename, param, parity, grid);
//   }
//   LatticeSet<T> _set;
//   _set.give(param.lattice_size, grid.lattice_size, parity);
//   _set.init();
//   _set._print();
//   cudaDeviceSynchronize();
//   cudaMalloc(&fermion_out_even, _set.lat_4dim_SC * _REAL_IMAG_ * sizeof(T));
//   cudaMalloc(&fermion_in_even, _set.lat_4dim_SC * _REAL_IMAG_ * sizeof(T));
//   cudaMalloc(&fermion_out_odd, _set.lat_4dim_SC * _REAL_IMAG_ * sizeof(T));
//   cudaMalloc(&fermion_in_odd, _set.lat_4dim_SC * _REAL_IMAG_ * sizeof(T));
//   cudaMalloc(&gauge, _set.lat_4dim_DCC * _REAL_IMAG_ * _EVEN_ODD_ * sizeof(T));
//   cudaDeviceSynchronize();
//   {   // test
//     { // io
//       std::stringstream filename;
//       filename << "wilson-clover-dslash-kappa1-fermion-out-all_1733904246_-16-16-32-32-524288-1-1-1-1-0-0-1-0-f.bin_";
//       device_load<T>(fermion_out_even, _set.lat_4dim_SC * _REAL_IMAG_, filename.str());
//       device_load<T>(fermion_out_odd, _set.lat_4dim_SC * _REAL_IMAG_, filename.str());
//     }
//     { // io
//       std::stringstream filename;
//       filename << "wilson-clover-dslash-kappa1-fermion-in-all_1733904246_-16-16-32-32-524288-1-1-1-1-0-0-1-0-f.bin_";
//       device_load<T>(fermion_in_even, _set.lat_4dim_SC * _REAL_IMAG_, filename.str());
//     }
//     { // io
//       std::stringstream filename;
//       filename << "wilson-clover-dslash-kappa1-gauge_1733904246_-16-16-32-32-524288-1-1-1-1-0-0-1-0-f.bin_";
//       device_load<T>(gauge, _set.lat_4dim_DCC * _REAL_IMAG_ * _EVEN_ODD_, filename.str());
//     }
//   }
//   { // -0-0-1-0-
//     dptzyxcc2ccdptzyx<T>(gauge, &_set);
//     tzyxsc2sctzyx<T>(fermion_in_even, &_set);
//     tzyxsc2sctzyx<T>(fermion_out_even, &_set);
//     {   // test
//       { // io
//         std::stringstream filename;
//         filename << "wilson-clover-dslash-kappa1-fermion-out-all_1733904246_-16-16-32-32-524288-1-1-1-1-0-0-1-0-f.bin";
//         device_save<T>(fermion_out_even, _set.lat_4dim_SC * _REAL_IMAG_, filename.str());
//       }
//       { // io
//         std::stringstream filename;
//         filename << "wilson-clover-dslash-kappa1-fermion-in-all_1733904246_-16-16-32-32-524288-1-1-1-1-0-0-1-0-f.bin";
//         device_save<T>(fermion_in_even, _set.lat_4dim_SC * _REAL_IMAG_, filename.str());
//       }
//       { // io
//         std::stringstream filename;
//         filename << "wilson-clover-dslash-kappa1-gauge_1733904246_-16-16-32-32-524288-1-1-1-1-0-0-1-0-f.bin";
//         device_save<T>(gauge, _set.lat_4dim_SC * _REAL_IMAG_, filename.str());
//       }
//     }
//   }
//   {     //-1-0-1-0-
//     {   // test
//       { // io
//         std::stringstream filename;
//         filename << "wilson-clover-dslash-kappa1-fermion-out-all_1733904246_-16-16-32-32-524288-1-1-1-1-0-0-1-0-f.bin_";
//         device_load<T>(fermion_out_even, _set.lat_4dim_SC * _REAL_IMAG_, filename.str());
//       }
//       { // io
//         std::stringstream filename;
//         filename << "wilson-clover-dslash-kappa1-fermion-in-all_1733904246_-16-16-32-32-524288-1-1-1-1-0-0-1-0-f.bin_";
//         device_load<T>(fermion_in_even, _set.lat_4dim_SC * _REAL_IMAG_, filename.str());
//       }
//       { // io
//         std::stringstream filename;
//         filename << "wilson-clover-dslash-kappa1-gauge_1733904246_-16-16-32-32-524288-1-1-1-1-0-0-1-0-f.bin_";
//         device_load<T>(gauge, _set.lat_4dim_DCC * _REAL_IMAG_ * _EVEN_ODD_, filename.str());
//       }
//     }
//     dptzyxcc2ccdptzyx<T>(gauge, &_set);
//     tzyxsc2sctzyx<T>(fermion_in_even, &_set);
//     tzyxsc2sctzyx<T>(fermion_out_even, &_set);
//     {   // test
//       { // io
//         std::stringstream filename;
//         filename << "wilson-clover-dslash-kappa1-fermion-out-all_1733904246_-16-16-32-32-524288-1-1-1-1-0-0-1-0-f.bin";
//         device_save<T>(fermion_out_even, _set.lat_4dim_SC * _REAL_IMAG_, filename.str());
//       }
//       { // io
//         std::stringstream filename;
//         filename << "wilson-clover-dslash-kappa1-fermion-in-all_1733904246_-16-16-32-32-524288-1-1-1-1-0-0-1-0-f.bin";
//         device_save<T>(fermion_in_even, _set.lat_4dim_SC * _REAL_IMAG_, filename.str());
//       }
//       { // io
//         std::stringstream filename;
//         filename << "wilson-clover-dslash-kappa1-gauge_1733904246_-16-16-32-32-524288-1-1-1-1-0-0-1-0-f.bin";
//         device_save<T>(gauge, _set.lat_4dim_SC * _REAL_IMAG_, filename.str());
//       }
//     }
//   }
//   _set.end();
//   cudaFree(gauge);
//   cudaFree(fermion_in_even);
//   cudaFree(fermion_out_even);
//   MPI_Finalize();
//   return 0;
// }