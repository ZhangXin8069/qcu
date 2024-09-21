#include "../include/qcu.h"
#include "define.h"
#ifdef LATTICE_SET
int give_flag(int count, ...) {
  int flag = 0;
  int _ = 1;
  va_list args;
  va_start(args, count);
  for (int i = 0; i < count; ++i) {
    flag += _ * va_arg(args, int);
    _ *= 2;
  }
  va_end(args);
  printf("###%d\n", flag);
  return  flag;
}
void get_flags(int _flag, int *flags) {
  int flag = _flag;
  for (int i = 0; i < _FLAGS_SIZE_; ++i) {
    flags[i] = flag / 2;
    flag -= flags[i] * 2;
    flag /= 2;
  }
}
#endif