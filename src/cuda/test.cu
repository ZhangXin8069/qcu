#include "qcu.h"
#include <iostream>
#include <cstdio>
#include <mpi.h>
#include "test_qcu_complex_computation.cuh"
QcuParam p;

class Debugger {
private:
    void* src;
    void* dst;
    void* gauge;
    int Lx_;
    int Ly_;
    int Lz_;
    int Lt_;
public:
    Debugger(int Lx, int Ly, int Lz, int Lt) : Lx_(Lx), Ly_(Ly), Lz_(Lz), Lt_(Lt) {
        int vol = Lx_ * Ly_ * Lz_ * Lt_;
        cudaMalloc(&dst, vol * 12 * sizeof(double) * 2);
        cudaMalloc(&src, vol * 12 * sizeof(double) * 2);
        cudaMalloc(&gauge, 4 * vol * 9 * sizeof(double) * 2);
    }
    ~Debugger() {
        cudaFree(dst);
        cudaFree(src);
        cudaFree(gauge);
    }
    void debug() {
        QcuGrid_t grid;

        grid.grid_size[0] = grid.grid_size[1] = grid.grid_size[2] = grid.grid_size[3] = 1;
        p.lattice_size[0] = 16;
        p.lattice_size[1] = 16;
        p.lattice_size[2] = 16;
        p.lattice_size[3] = 32;
        initGridSize(&grid, &p, gauge, src, dst);
        // void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity);
        int parity = 0;
        double* fermion_in = static_cast<double*>(src) + (1-parity) * Lx_ * Ly_ * Lz_ * Lt_ * 12 * 2 / 2;
        double* fermion_out = static_cast<double*>(src) + parity * Lx_ * Ly_ * Lz_ * Lt_ * 12 * 2 / 2;
        dslashQcu(static_cast<void*>(fermion_out), static_cast<void*>(fermion_in), gauge, &p, parity);
        parity = 1;
        fermion_in = static_cast<double*>(src) + (1-parity) * Lx_ * Ly_ * Lz_ * Lt_ * 12 * 2 / 2;
        fermion_out = static_cast<double*>(src) + parity * Lx_ * Ly_ * Lz_ * Lt_ * 12 * 2 / 2;
        dslashQcu(static_cast<void*>(fermion_out), static_cast<void*>(fermion_in), gauge, &p, parity);
    }
};

int main (int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Debugger debugger(32, 32, 32, 64);
    debugger.debug();
    MPI_Finalize();
    //cg_inverter();
    //test_computation();
    return 0;
}

