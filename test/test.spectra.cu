#include <Eigen/Dense>
#include <Spectra/SparseSymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <iostream>
#include <complex>
#include <chrono>

using namespace Spectra;
using namespace Eigen;
using namespace std;

using T = float;
using ComplexT = complex<T>;

int main()
{
    Eigen::Index n = 1000;
    Eigen::SparseMatrix<ComplexT> mat(n, n);

    // Initialize a random sparse matrix (Hermitian)
    srand(time(0));
    for (Eigen::Index i = 0; i < n; i++) {
        for (Eigen::Index j = i; j < n; j++) {
            if (rand() % 10 < 3) {  // Sparse (30% non-zero entries)
                ComplexT value = ComplexT(rand() % 10 + 1, rand() % 10 + 1);
                mat.insert(i, j) = value;
                mat.insert(j, i) = value; // Ensure Hermitian symmetry
            }
        }
    }

    // Create SparseSymMatProd operator
    SparseSymMatProd<ComplexT> op_sparse(mat); 

    // Specify the number of eigenvalues and the subspace size
    Eigen::Index num_of_eigenvalues = 400;
    SparseSymEigsSolver<SparseSymMatProd<ComplexT>> solver(op_sparse, num_of_eigenvalues, num_of_eigenvalues * 2); 

    // Solver settings
    Eigen::Index iterations = 100;
    T tolerance = 1e-6;

    // Time the solver
    auto start = chrono::high_resolution_clock::now();
    int nconv = solver.compute(SortRule::LargestAlge, iterations, tolerance);
    auto end = chrono::high_resolution_clock::now();
    cout << "SparseSymEigsSolver: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

    // Retrieve and print the eigenvalues
    if (solver.info() == CompInfo::Successful)
    {
        Eigen::VectorXcd evalues = solver.eigenvalues();
        cout << nconv << " Eigenvalues found:\n";
        for (int i = 0; i < 10; i++) {
            cout << evalues[i] << endl;
        }
    }
    else
    {
        cout << "Calculation failed" << endl;
    }

    return 0;
}
