#include <iostream>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <chrono>
#include <vector>
#include <random>
#include <cmath>

using namespace Eigen;
using namespace std;
using Complex = std::complex<double>;
using ComplexVector = Matrix<Complex, Dynamic, 1>;
using ComplexMatrix = Matrix<Complex, Dynamic, Dynamic>;

// Utility function for safe matrix multiplication
ComplexMatrix safe_matrix_mult(const ComplexMatrix &A, const ComplexMatrix &B)
{
    if (A.cols() != B.rows())
    {
        throw std::runtime_error("Matrix multiplication dimensions do not match");
    }
    return A * B;
}

// Utility function for safe matrix-vector multiplication
ComplexVector safe_matrix_vector_mult(const ComplexMatrix &A, const ComplexVector &v)
{
    if (A.cols() != v.rows())
    {
        throw std::runtime_error("Matrix-vector multiplication dimensions do not match");
    }
    return A * v;
}

// Utility function for custom diagonal extraction
template <typename Derived>
Matrix<typename Derived::Scalar, Dynamic, Dynamic>
custom_diag(const Eigen::MatrixBase<Derived> &input)
{
    if (input.rows() == input.cols())
    {
        return input.diagonal().asDiagonal();
    }
    else if (input.rows() == 1 || input.cols() == 1)
    {
        return input.derived().asDiagonal();
    }
    throw std::runtime_error("Invalid input for diagonal extraction");
}

// Custom QR decomposition for complex matrices
void custom_qr_decomp(const ComplexMatrix &A,
                      ComplexMatrix &Q,
                      ComplexMatrix &R,
                      double tol = 1e-9)
{
    int m = A.rows();
    int n = A.cols();
    Q = ComplexMatrix::Zero(m, n);
    R = ComplexMatrix::Zero(n, n);

    for (int j = 0; j < n; ++j)
    {
        ComplexVector v = A.col(j);
        for (int i = 0; i < j; ++i)
        {
            R(i, j) = Q.col(i).conjugate().dot(v);
            v -= R(i, j) * Q.col(i);
        }
        R(j, j) = v.norm();
        if (std::abs(R(j, j)) > tol)
        {
            Q.col(j) = v / R(j, j);
        }
    }
}

// Orthonormalization
ComplexMatrix custom_orth(const ComplexMatrix &A, double tol = 1e-9)
{
    ComplexMatrix Q, R;
    custom_qr_decomp(A, Q, R);

    int rank = 0;
    for (int i = 0; i < R.diagonal().size(); ++i)
    {
        if (std::abs(R.diagonal()[i]) > tol)
        {
            ++rank;
        }
    }
    return Q.leftCols(rank);
}

// Generate sparse matrix with specified eigenvalues
ComplexMatrix generate_sparse_matrix(const ComplexVector &eigenvalues,
                                     const ComplexMatrix &eigenvectors)
{
    int n = eigenvalues.size();
    ComplexMatrix D = ComplexMatrix::Zero(n, n);
    D.diagonal() = eigenvalues;

    return safe_matrix_mult(safe_matrix_mult(eigenvectors, D), eigenvectors.conjugate().transpose());
}

// Chebyshev polynomial application
ComplexVector apply_chebyshev_polynomial(const ComplexMatrix &A,
                                         const ComplexVector &v,
                                         int degree)
{
    vector<ComplexVector> T(degree + 1);
    T[0] = v;
    T[1] = safe_matrix_vector_mult(A, v);

    for (int k = 2; k <= degree; ++k)
    {
        T[k] = 2 * safe_matrix_vector_mult(A, T[k - 1]) - T[k - 2];
    }
    return T[degree];
}

// Arnoldi iteration with dimension safety
void arnoldi_iteration(const ComplexMatrix &A,
                       int k,
                       const ComplexVector &v,
                       ComplexMatrix &Q,
                       ComplexMatrix &H)
{
    int n = A.rows();
    k = min(k, n - 1); // Ensure k doesn't exceed matrix dimensions
    Q = ComplexMatrix::Zero(n, k + 1);
    H = ComplexMatrix::Zero(k + 1, k);

    // Normalize initial vector
    ComplexVector q = v / v.norm();
    Q.col(0) = q;

    for (int j = 0; j < k; ++j)
    {
        ComplexVector w = safe_matrix_vector_mult(A, Q.col(j));

        // Orthogonalization
        for (int i = 0; i <= j; ++i)
        {
            H(i, j) = Q.col(i).conjugate().dot(w);
            w -= H(i, j) * Q.col(i);
        }

        // Normalize
        H(j + 1, j) = w.norm();
        if (std::abs(H(j + 1, j)) > 1e-10)
        {
            Q.col(j + 1) = w / H(j + 1, j);
        }
    }
}

// QR Algorithm with improved convergence check
pair<ComplexVector, ComplexMatrix> qr_algorithm(ComplexMatrix H,
                                                int max_iter = 5000,
                                                double tol = 1e-9)
{
    ComplexMatrix Q, R;
    for (int iter = 0; iter < max_iter; ++iter)
    {
        custom_qr_decomp(H, Q, R);
        H = safe_matrix_mult(R, Q.leftCols(H.cols()));

        bool converged = true;
        for (int i = 0; i < H.rows(); ++i)
        {
            for (int j = 0; j < H.cols(); ++j)
            {
                if (i != j && std::abs(H(i, j)) >= tol)
                {
                    converged = false;
                    break;
                }
            }
            if (!converged)
                break;
        }

        if (converged)
        {
            cout << "Converged in " << iter << " iterations" << endl;
            break;
        }
    }

    ComplexVector eigenvalues = H.diagonal();
    return {eigenvalues, Q};
}

// Eigenvalue validation with more robust checks
void validate_eigenvector(const ComplexMatrix &A,
                          const ComplexVector &eigenvalues,
                          const ComplexVector &eigenvector)
{
    if (eigenvector.size() == 0)
    {
        cout << "Empty eigenvector" << endl;
        return;
    }

    // Normalize eigenvector
    ComplexVector normalized_vec = eigenvector / eigenvector.norm();

    ComplexVector Av = safe_matrix_vector_mult(A, normalized_vec);
    ComplexVector lambda_v = eigenvalues(0) * normalized_vec;

    double norm_Av = Av.norm();
    double norm_lambda_v = lambda_v.norm();
    double error = (Av - lambda_v).norm() / (norm_Av + 1e-10);

    cout << "Eigenvector norm: " << normalized_vec.norm() << endl;
    cout << "norm_Av: " << norm_Av << endl;
    cout << "norm_lambda_v: " << norm_lambda_v << endl;
    cout << "Error: " << error << endl;
}

int main()
{
    const int N = 1000;
    const int n = 50;
    const int degree = 5;
    const int k = 60;
    const int max_iter = 10000;

    // Random number generation with consistent seed for reproducibility
    mt19937 gen(42); // Fixed seed
    normal_distribution<> dis(0.0, 1.0);

    // Generate eigenvalues
    ComplexVector eigenvalues = ComplexVector::Zero(N);
    for (int i = 0; i < n; ++i)
    {
        eigenvalues(i) = Complex(i * 0.5, i * 0.5);
    }

    // Generate random eigenvectors with more controlled generation
    ComplexMatrix random_matrix(N, N);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            random_matrix(i, j) = Complex(dis(gen), dis(gen));
        }
    }

    ComplexMatrix eigenvectors = custom_orth(random_matrix);
    ComplexMatrix A = generate_sparse_matrix(eigenvalues, eigenvectors);

    // Ensure matrix is Hermitian (for better numerical stability)
    A = (A + A.conjugate().transpose()) / 2.0;

    // Generate random vector
    ComplexVector v = ComplexVector::Zero(N);
    for (int i = 0; i < N; ++i)
    {
        v(i) = Complex(dis(gen), dis(gen));
    }
    v = v / v.norm(); // Normalize initial vector

    // Method 1: Eigen's built-in eigenvalue computation
    auto start = chrono::high_resolution_clock::now();
    ComplexMatrix B = A; // Use a copy to avoid modifying original matrix
    Eigen::ComplexEigenSolver<ComplexMatrix> ces(B);
    auto end = chrono::high_resolution_clock::now();

    cout << "Eigen method time: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    validate_eigenvector(A, ces.eigenvalues(), ces.eigenvectors().col(0));

    // Method 2: Arnoldi + QR
    start = chrono::high_resolution_clock::now();
    ComplexMatrix Q, H;
    arnoldi_iteration(A, k, v, Q, H);
    auto [arnoldi_eigenvalues, arnoldi_eigenvectors] = qr_algorithm(H, max_iter);
    ComplexVector final_eigenvector = Q.leftCols(k) * arnoldi_eigenvectors.col(0); // Use leftCols(k)
    end = chrono::high_resolution_clock::now();

    cout << "Arnoldi + QR time: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    validate_eigenvector(A, arnoldi_eigenvalues, final_eigenvector);

    // Method 3: Chebyshev + Arnoldi + QR
    start = chrono::high_resolution_clock::now();
    ComplexVector v_chebyshev = apply_chebyshev_polynomial(A, v, degree);
    arnoldi_iteration(A, k, v_chebyshev, Q, H);
    auto [cheb_eigenvalues, cheb_eigenvectors] = qr_algorithm(H, max_iter);
    ComplexVector final_cheb_eigenvector = Q.leftCols(k) * cheb_eigenvectors.col(0); // Use leftCols(k)
    end = chrono::high_resolution_clock::now();

    cout << "Chebyshev + Arnoldi + QR time: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    validate_eigenvector(A, cheb_eigenvalues, final_cheb_eigenvector);

    return 0;
}
