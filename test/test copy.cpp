#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <complex>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace Eigen;
using namespace std;

class CustomMultiplier
{
private:
    MatrixXcd original_matrix;
    int n;

public:
    CustomMultiplier(int size) : n(size)
    {
        VectorXcd eigenvalues(size);
        MatrixXcd V = MatrixXcd::Random(size, size);

        for (int i = 0; i < size; i++)
        {
            double real = cos(2.0 * M_PI * i / size);
            double imag = sin(2.0 * M_PI * i / size);
            eigenvalues(i) = complex<double>(real, imag);
        }

        HouseholderQR<MatrixXcd> qr(V);
        MatrixXcd Q = qr.householderQ();
        original_matrix = Q * eigenvalues.asDiagonal() * Q.adjoint();
    }

    VectorXcd operator*(const VectorXcd &vec) const
    {
        return original_matrix * vec;
    }

    const MatrixXcd &getMatrix() const { return original_matrix; }
};

class ArnoldiSolver
{
private:
    int n, m;
    const CustomMultiplier &multiplier;

    complex<double> computeRayleighQuotient(const VectorXcd &v)
    {
        VectorXcd Av = multiplier * v;
        return (v.adjoint() * Av)(0, 0) / (v.adjoint() * v)(0, 0);
    }

    VectorXcd refineSingleVector(const complex<double> &initial_eigenvalue,
                                 const VectorXcd &initial_vector,
                                 complex<double> &refined_eigenvalue,
                                 int max_iter = 5)
    {
        VectorXcd current_vector = initial_vector;
        refined_eigenvalue = initial_eigenvalue;

        for (int iter = 0; iter < max_iter; ++iter)
        {
            VectorXcd new_vector = multiplier * current_vector;
            refined_eigenvalue = computeRayleighQuotient(current_vector);
            new_vector.normalize();

            if ((new_vector - current_vector).norm() < 1e-10)
            {
                break;
            }
            current_vector = new_vector;
        }
        return current_vector;
    }

public:
    ArnoldiSolver(int size, int num_eigenvalues, const CustomMultiplier &mult)
        : n(size), m(num_eigenvalues), multiplier(mult) {}

    void solve(vector<complex<double>> &eigenvalues, MatrixXcd &eigenvectors)
    {
        MatrixXcd V = MatrixXcd::Zero(n, m + 1);
        MatrixXcd H = MatrixXcd::Zero(m + 1, m);

        V.col(0) = VectorXcd::Random(n).normalized();

        for (int j = 0; j < m; j++)
        {
            VectorXcd w = multiplier * V.col(j);

            for (int i = 0; i <= j; i++)
            {
                complex<double> h = V.col(i).adjoint() * w;
                H(i, j) = h;
                w -= h * V.col(i);
            }

            double norm_w = w.norm();
            if (norm_w < 1e-10)
                break;

            H(j + 1, j) = norm_w;
            V.col(j + 1) = w / norm_w;
        }

        MatrixXcd H_m = H.topLeftCorner(m, m);
        ComplexEigenSolver<MatrixXcd> ces(H_m);

        eigenvalues.resize(m);
        eigenvectors = MatrixXcd::Zero(n, m);

        for (int i = 0; i < m; i++)
        {
            eigenvalues[i] = ces.eigenvalues()(i);
            VectorXcd initial_vector = V.leftCols(m) * ces.eigenvectors().col(i);
            initial_vector.normalize();

            complex<double> refined_eigenvalue;
            VectorXcd refined_vector = refineSingleVector(eigenvalues[i],
                                                          initial_vector,
                                                          refined_eigenvalue);

            eigenvalues[i] = refined_eigenvalue;
            eigenvectors.col(i) = refined_vector;
        }
    }
};

void verify_results(const CustomMultiplier &multiplier,
                    const vector<complex<double>> &computed_eigenvalues,
                    const MatrixXcd &computed_eigenvectors)
{
    cout << "Verification Results:" << endl;

    vector<pair<double, int>> errors(computed_eigenvalues.size());

    for (size_t i = 0; i < computed_eigenvalues.size(); i++)
    {
        VectorXcd eigenvector = computed_eigenvectors.col(i);
        VectorXcd Av = multiplier * eigenvector; // Using custom multiplier here
        VectorXcd lambda_v = computed_eigenvalues[i] * eigenvector;

        double error = (Av - lambda_v).norm() / eigenvector.norm();
        errors[i] = make_pair(error, i);

        cout << "Eigenvalue " << i + 1 << ": " << computed_eigenvalues[i] << endl;
        cout << "Relative Error: " << error << endl;
        cout << "-------------------" << endl;
    }

    sort(errors.begin(), errors.end());
    cout << "\nBest 5 Results:" << endl;
    for (int i = 0; i < min(5, (int)errors.size()); ++i)
    {
        cout << "Eigenvalue " << errors[i].second + 1
             << " Relative Error: " << errors[i].first << endl;
    }
}

int main()
{
    const int matrix_size = 100;
    const int num_eigenvalues = 50;

    CustomMultiplier multiplier(matrix_size);
    ArnoldiSolver solver(matrix_size, num_eigenvalues, multiplier);

    vector<complex<double>> eigenvalues;
    MatrixXcd eigenvectors;
    solver.solve(eigenvalues, eigenvectors);

    verify_results(multiplier, eigenvalues, eigenvectors);

    return 0;
}