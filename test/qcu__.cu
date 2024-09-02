#include "./include/qcu.h"
#include <nccl.h>
int main() {
  MPI_Init(NULL, NULL);
  int param_lattice_size[_DIM_];
  int grid_lattice_size[_DIM_];
  for (int i = 0; i < _DIM_; i++) {
    param_lattice_size[i] = _LAT_EXAMPLE_;
    grid_lattice_size[i] = _GRID_EXAMPLE_ * 2;
  }
  grid_lattice_size[_T_] = 1;
  LatticeSet _set;
  _set.give(param_lattice_size, grid_lattice_size);
  _set.init();
  _set.end();
  MPI_Finalize();
  return 0;
}