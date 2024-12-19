#include <iostream>
#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <random>
#include <chrono>

using namespace Eigen;
using namespace std;

// Helper function to compute diagonal matrix
template <typename MatrixType>
MatrixType np_diag(const MatrixType &input)
{
    if (input.rows() == 1)
    {
        int n = input.cols();
        MatrixType diag_matrix = MatrixType::Zero(n, n);
        for (int i = 0; i < n; ++i)
        {
            diag_matrix(i, i) = input(i);
        }
        return diag_matrix;
    }
    else
    {
        return input.diagonal().asDiagonal();
    }
}

// Chebyshev Polynomial method
template <typename MatrixType>
VectorXcd applyChebyshevPolynomial(const MatrixType &A, const VectorXcd &v, int degree)
{
    VectorXcd T = VectorXcd::Zero(A.rows(), degree + 1);  // Initialize a vector of size degree + 1
    T.col(0) = v;                                           // Store vector v in T(0)
    T.col(1) = A * v;                                       // Compute A * v and store in T(1)

    for (int k = 2; k <= degree; ++k)
    {
        T.col(k) = 2.0 * A * T.col(k - 1) - T.col(k - 2);  // Chebyshev recurrence relation
    }
    return T.col(degree); // Return the value at degree
}

// Arnoldi Iteration
template <typename MatrixType>
void arnoldiIteration(const MatrixType &A, int k, const VectorXcd &v, MatrixXcd &Q, MatrixXcd &H)
{
    int n = A.rows();
    VectorXcd q = v.normalized();
    Q.col(0) = q;
    for (int j = 0; j < k; ++j)
    {
        VectorXcd w = A * Q.col(j); // Dslash operation
        for (int i = 0; i <= j; ++i)
        {
            H(i, j) = Q.col(i).adjoint() * w; // Compute H(i, j)
            w -= H(i, j) * Q.col(i);          // Subtract projection from w
        }
        H(j + 1, j) = w.norm();  // Norm of the residual
        Q.col(j + 1) = w / H(j + 1, j);  // Normalize the new vector
    }
}

// QR Decomposition with Eigenvalues extraction using HouseholderQR
template <typename MatrixType>
std::pair<VectorXcd, MatrixXcd> qrAlgorithm(const MatrixType &H, int max_iter = 5000, double tol = 1e-9)
{
    MatrixXcd H_current = H;
    MatrixXcd Q = MatrixXcd::Identity(H.rows(), H.cols());

    for (int i = 0; i < max_iter; ++i)
    {
        // Perform QR decomposition using HouseholderQR
        HouseholderQR<MatrixXcd> qr(H_current);
        MatrixXcd qr_matrix = qr.matrixQR(); // Get the combined matrix

        // Extract Q and R from the combined matrix
        MatrixXcd Q_new = qr_matrix.leftCols(H_current.cols());
        MatrixXcd R_new = qr_matrix.rightCols(H_current.cols());

        // Update H_current and Q
        H_current = Q_new.adjoint() * R_new;
        Q = Q * Q_new;  // Accumulate the Q factors

        // Extract diagonal matrix from H_current
        MatrixXcd H_diag = H_current.diagonal().asDiagonal();

        // Check for convergence (when off-diagonal elements are small enough)
        if ((H_current - H_diag).norm() < tol)
        {
            cout << "Converged after " << i << " iterations." << endl;
            break;
        }
    }

    // Return the diagonal (eigenvalues) and eigenvectors (Q)
    return std::make_pair(H_current.diagonal(), Q);
}

// Generate sparse matrix from eigenvalues and eigenvectors
MatrixXcd generateSparseMatrix(int n, const VectorXcd &eigenvalues)
{
    MatrixXcd D = eigenvalues.asDiagonal();
    MatrixXcd V = MatrixXcd::Random(n, n);
    JacobiSVD<MatrixXcd> svd(V, ComputeThinU | ComputeThinV);
    V = svd.matrixU();
    return V * D * V.adjoint();
}

// Main function
int main()
{
    int N = 1000, n = 50;
    VectorXcd eigenvalues(n);
    for (int i = 0; i < n; ++i)
    {
        eigenvalues(i) = complex<double>(i * 0.5, i * 0.5);
    }

    // Generate a random eigenvector matrix for sparse matrix A
    MatrixXcd A = generateSparseMatrix(N, eigenvalues);

    // Choose Krylov subspace size and degree for Chebyshev
    int k = 60;     // Krylov subspace dimension
    int degree = 5; // Degree for Chebyshev polynomial

    // Step 1: Apply Chebyshev polynomial to A and vector v
    VectorXcd v = VectorXcd::Random(N);
    VectorXcd v_chebyshev = applyChebyshevPolynomial(A, v, degree);

    // Step 2: Arnoldi Iteration
    MatrixXcd Q = MatrixXcd::Zero(N, k + 1);
    MatrixXcd H = MatrixXcd::Zero(k + 1, k);
    arnoldiIteration(A, k, v_chebyshev, Q, H);

    // Step 3: QR Algorithm on H using HouseholderQR
    auto [eigenvalues_result, eigenvectors] = qrAlgorithm(H);

    // Step 4: Compute eigenvectors
    MatrixXcd final_eigenvectors = Q.leftCols(k) * eigenvectors;

    // Output the results
    cout << "Eigenvalues: " << eigenvalues_result.transpose() << endl;
    cout << "Eigenvectors (first few): " << final_eigenvectors.leftCols(3) << endl; // Display first few eigenvectors

    return 0;
}
