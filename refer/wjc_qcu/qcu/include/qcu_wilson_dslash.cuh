#pragma once

#include "qcu_dslash.cuh"
// class WilsonDslash : public Dslash {
// public:
//   WilsonDslash(DslashParam& param) : Dslash(param){}
//   virtual void calculateDslash(int invert_flag = 0);
// };
// mpiDslash
class MpiWilsonDslash : public Dslash {
public:
  MpiWilsonDslash(DslashParam& param) : Dslash(param){}
  virtual void calculateDslash(int invert_flag = 0);
};

// void callWilsonDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity);
void callMpiWilsonDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag);

void fullWilsonDslashOneRound (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int invert_flag);
void wilsonDslashOneRound(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int invert_flag);