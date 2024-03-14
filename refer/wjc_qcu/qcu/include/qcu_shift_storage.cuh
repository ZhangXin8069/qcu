#pragma once

enum MemoryStorage {
  NON_COALESCED = 0,
  COALESCED = 1,
};

enum ShiftDirection {
  TO_COALESCE = 0,
  TO_NON_COALESCE = 1,
};


void shiftVectorStorage(void* dst_vec, void* src_vec, int shift_direction, int Lx, int Ly, int Lz, int Lt);
// void shiftGaugeStorage(void* dst_vec, void* src_vec, int shift_direction, int Lx, int Ly, int Lz, int Lt);
void shiftGaugeStorage(void* dst_vec, void* src_vec, int shift_direction, int Lx, int Ly, int Lz, int Lt);