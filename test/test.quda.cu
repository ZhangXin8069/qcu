#include "../include/qcu.h"
#include <quda.h>
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  unsigned long long seed(666);
  double sigma(0.1);
  gaussGaugeQuda(seed, sigma);
  MPI_Finalize();
  return 0;
}