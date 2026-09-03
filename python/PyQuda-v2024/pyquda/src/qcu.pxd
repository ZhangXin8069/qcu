cdef extern from "qcu.h":
    ctypedef struct QcuParam:
        int lattice_size[4]
    void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity)
    void dslashCloverQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity)
    void mpiDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, QcuParam *grid)
    void mpiBistabCgQcu(void *gauge, QcuParam *param, QcuParam *grid)
    void ncclDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, QcuParam *grid)
    void ncclBistabCgQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, QcuParam *grid)
    void ncclDslashCloverQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, QcuParam *grid)