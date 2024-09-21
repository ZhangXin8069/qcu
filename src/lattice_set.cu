#include "../include/qcu.h"
#include "define.h"
#ifdef LATTICE_SET
void give_flag(int flag, ...) {
  flag = 0;
  int _ = 1;
  va_list args;
  va_start(args, _FLAGS_SIZE_);
  for (int i = 0; i < _FLAGS_SIZE_; ++i) {
    flag += _ * va_arg(args, int);
    _ *= 2;
  }
  va_end(args);
}
void get_flags(int _flag, int *flags) {
  int flag = _flag;
  int _ = 1;
  for (int i = 0; i < _FLAGS_SIZE_; ++i) {
    flags[i] = flag % 2;
    _ *= 2;
    flag -= flags[i] * 2;
    flag /= 2;
  }
}
#endif