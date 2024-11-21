cdef extern from "qcu.h":
    ctypedef struct QcuParam:
        int lattice_size[4]
    void testDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity)
    void testCloverDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity)
    void mpiDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, QcuParam *grid)
    void mpiBistabCgQcu(void *gauge, QcuParam *param, QcuParam *grid)
    void applyDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, QcuParam *grid)
    void applyBistabCgQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, QcuParam *grid)
    void applyCgQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, QcuParam *grid)
    void applyCloverDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, QcuParam *grid)