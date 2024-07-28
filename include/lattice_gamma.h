#ifndef _LATTICE_GAMMA_H
#define _LATTICE_GAMMA_H

#include "./qcu.h"

// Lattice gamma class
class LatticeGamma {
public:
  // Constructors

  // Default constructor
  __host__ __device__ LatticeGamma() {}

/*
$\gamma_0 = \begin{pmatrix} 0 & 0 & 0 & i \\ 0 & 0 & i & 0 \\ 0 & -i & 0 & 0 \\ -i & 0 & 0 & 0 \\ \end {pmatrix}$

$\gamma_1 = \begin{pmatrix} 0 & 0 & 0 & -1 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ -1 & 0 & 0 & 0 \\ \end {pmatrix}$

$\gamma_2 = \begin{pmatrix} 0 & 0 & i & 0 \\ 0 & 0 & 0 & -i \\ -i & 0 & 0 & 0 \\ 0 & i & 0 & 0 \\ \end {pmatrix}$

$\gamma_3 = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ \end {pmatrix}$
*/

  // Destructor
  ~LatticeGamma() {}

};

#endif