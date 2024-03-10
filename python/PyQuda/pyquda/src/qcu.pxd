cdef extern from "qcu.h":
    ctypedef struct QcuParam:
        int lattice_size[4]
    void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity)
    void mpiDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, QcuParam *grid)
    void mpiCgQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, QcuParam *grid)
    void testDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity)