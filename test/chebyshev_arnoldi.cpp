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
VectorXcd applyChebyshevPolynomial(const MatrixType& A, const VectorXcd& v, int degree) {
    VectorXcd T = VectorXcd::Zero(A.rows(), degree + 1);  // 初始化一个大小为 degree + 1 的零向量
    T(0) = v;  // 将向量 v 存入 T 的第一个位置
    T(1) = A * v;  // 计算 A*v 并存储在 T(1)

    for (int k = 2; k <= degree; ++k) {
        T(k) = 2.0 * A * T(k - 1) - T(k - 2);
    }

    return T(degree);  // 返回第 degree 个值
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
        VectorXcd w = A * Q.col(j);
        for (int i = 0; i <= j; ++i)
        {
            H(i, j) = Q.col(i).adjoint() * w;
            w -= H(i, j) * Q.col(i);
        }
        H(j + 1, j) = w.norm();
        if (H(j + 1, j) != std::complex<double>(0, 0))
        { // 修复这里
            Q.col(j + 1) = w / H(j + 1, j);
        }
    }
}

// QR Decomposition with Eigenvalues extraction
template <typename MatrixType>
std::pair<VectorXcd, MatrixXcd> qrAlgorithm(const MatrixType& H, int max_iter = 5000, double tol = 1e-9) {
    MatrixXcd H_current = H;
    MatrixXcd Q = MatrixXcd::Identity(H.rows(), H.cols());

    for (int i = 0; i < max_iter; ++i) {
        // Perform QR decomposition
        JacobiSVD<MatrixXcd> svd(H_current, ComputeThinU | ComputeThinV);
        H_current = svd.matrixV() * svd.matrixU().adjoint();
        Q = Q * svd.matrixU();

        // Extract the diagonal matrix of H_current
        MatrixXcd H_diag = H_current.diagonal().asDiagonal();

        // Compare the difference between H_current and the diagonal of H_current
        if ((H_current - H_diag).norm() < tol) {
            std::cout << "Converged after " << i << " iterations." << std::endl;
            break;
        }
    }

    // Return the diagonal of H_current (eigenvalues) and Q (eigenvectors)
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

    // Step 3: QR Algorithm on H
    auto [eigenvalues_result, eigenvectors] = qrAlgorithm(H);

    // Step 4: Compute eigenvectors
    MatrixXcd final_eigenvectors = Q.leftCols(k) * eigenvectors;

    // Output the results
    cout << "Eigenvalues: " << eigenvalues_result.transpose() << endl;
    cout << "Eigenvectors: " << final_eigenvectors.leftCols(3) << endl; // Display first few eigenvectors

    return 0;
}
