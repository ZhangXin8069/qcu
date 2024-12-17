#include <iostream>
#include <complex>
#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace Eigen;
using namespace std;
using Complex = std::complex<double>;
using ComplexMatrix = Matrix<Complex, Dynamic, Dynamic>;
using ComplexVector = Matrix<Complex, Dynamic, 1>;

// Utility function to create diagonal matrix
ComplexMatrix _diag(const ComplexVector &input)
{
    return input.asDiagonal();
}

ComplexVector _diag(const ComplexMatrix &input)
{
    return input.diagonal();
}

// QR decomposition function (simplified)
void _qr(const ComplexMatrix &A, ComplexMatrix &Q, ComplexMatrix &R)
{
    HouseholderQR<ComplexMatrix> qr(A);
    Q = qr.householderQ();
    R = Q.inverse() * A;
}

// Orthogonalization function
ComplexMatrix _orth(const ComplexMatrix &A, double tol = 1e-9)
{
    ComplexMatrix Q, R;
    _qr(A, Q, R);

    int rank = 0;
    for (int i = 0; i < R.rows(); ++i)
    {
        if (abs(R(i, i)) > tol)
        {
            rank++;
        }
    }

    return Q.leftCols(rank);
}

// Generate sparse matrix with given eigenvalues and eigenvectors
ComplexMatrix generate_sparse_matrix_complex(
    const ComplexVector &eigenvalues,
    const ComplexMatrix &eigenvectors)
{
    ComplexMatrix D = eigenvalues.asDiagonal();
    return eigenvectors * D * eigenvectors.adjoint();
}

// Arnoldi iteration
void arnoldi_iteration_complex(
    const ComplexMatrix &A,
    int k,
    const ComplexVector &v,
    ComplexMatrix &Q,
    ComplexMatrix &H)
{
    int n = A.rows();
    Q = ComplexMatrix::Zero(n, k + 1);
    H = ComplexMatrix::Zero(k + 1, k);

    ComplexVector q = v / v.norm();
    Q.col(0) = q;

    for (int j = 0; j < k; ++j)
    {
        ComplexVector w = A * Q.col(j);
        for (int i = 0; i <= j; ++i)
        {
            H(i, j) = Q.col(i).adjoint() * w;
            w -= H(i, j) * Q.col(i);
        }
        H(j + 1, j) = w.norm();

        // Use std::abs to compare complex number magnitude
        if (std::abs(H(j + 1, j)) > 1e-10 && j + 1 < n)
        {
            Q.col(j + 1) = w / H(j + 1, j);
        }
    }
}

// Validation function
void validate_eigenvector_complex(
    const ComplexMatrix &A,
    const ComplexVector &eigenvalues,
    const ComplexMatrix &eigenvectors)
{
    Complex eigenvalue = eigenvalues(0);
    ComplexVector eigenvector = eigenvectors.col(0);

    ComplexVector Av = A * eigenvector;
    ComplexVector lambda_v = eigenvalue * eigenvector;

    double norm_Av = Av.norm();
    double norm_lambda_v = lambda_v.norm();
    double error = (Av - lambda_v).norm() / norm_Av;

    cout << "Eigenvector norm: " << eigenvector.norm() << endl;
    cout << "norm_Av: " << norm_Av << endl;
    cout << "norm_lambda_v: " << norm_lambda_v << endl;
    cout << "Error: " << error << endl;
}

// Custom complex eigenvalue computation
class CustomComplexEigenSolver {
private:
    ComplexMatrix m_matrix;
    ComplexVector m_eigenvalues;
    ComplexMatrix m_eigenvectors;

public:
    CustomComplexEigenSolver(const ComplexMatrix& matrix) : m_matrix(matrix) {
        compute();
    }

    void compute() {
        // Use Eigen's EigenSolver as a base
        EigenSolver<ComplexMatrix> es(m_matrix);
        
        // Manually copy eigenvalues
        int n = m_matrix.rows();
        m_eigenvalues = ComplexVector(n);
        for (int i = 0; i < n; ++i) {
            m_eigenvalues(i) = Complex(
                es.eigenvalues()(i).real(), 
                es.eigenvalues()(i).imag()
            );
        }

        // Manually copy eigenvectors
        m_eigenvectors = ComplexMatrix(n, n);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                m_eigenvectors(i, j) = Complex(
                    es.eigenvectors().coeff(i, j).real(),
                    es.eigenvectors().coeff(i, j).imag()
                );
            }
        }
    }

    const ComplexVector& eigenvalues() const { return m_eigenvalues; }
    const ComplexMatrix& eigenvectors() const { return m_eigenvectors; }
};

int main()
{
    const int N = 1000;
    const int n = 50;
    const int degree = 5;
    const int k = 60;

    // Create random eigenvalues
    ComplexVector eigenvalues(N);
    for (int i = 0; i < n; ++i)
    {
        eigenvalues(i) = Complex(i * 0.5, i * 0.5);
    }
    for (int i = n; i < N; ++i)
    {
        eigenvalues(i) = 0;
    }

    // Generate random matrix
    ComplexMatrix random_matrix = ComplexMatrix::Random(N, N);
    ComplexMatrix eigenvectors = _orth(random_matrix);
    ComplexMatrix A = generate_sparse_matrix_complex(eigenvalues, eigenvectors);

    // Random initial vector
    ComplexVector v = ComplexVector::Random(N);

    // Method 1: Direct Eigen Eigenvalue Computation
    auto start = chrono::high_resolution_clock::now();

    CustomComplexEigenSolver es(A);
    ComplexVector eigen_values = es.eigenvalues();
    ComplexMatrix eigen_vectors = es.eigenvectors();

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;

    cout << "Eigen Solver time: " << diff.count() << " seconds" << endl;
    validate_eigenvector_complex(A, eigen_values, eigen_vectors);

    // Method 2: Arnoldi + Eigen Solver
    start = chrono::high_resolution_clock::now();

    ComplexMatrix Q, H;
    arnoldi_iteration_complex(A, k, v, Q, H);

    CustomComplexEigenSolver qr_solver(H);
    ComplexVector arnoldi_values = qr_solver.eigenvalues();
    ComplexMatrix arnoldi_vectors = qr_solver.eigenvectors();

    ComplexMatrix final_vectors = Q * arnoldi_vectors;

    end = chrono::high_resolution_clock::now();
    diff = end - start;

    cout << "Arnoldi + Eigen Solver time: " << diff.count() << " seconds" << endl;
    validate_eigenvector_complex(A, arnoldi_values, final_vectors);

    return 0;
}