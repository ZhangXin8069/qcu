#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <complex>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace Eigen;
using namespace std;
class ChebyshevAccelerator
{
private:
    vector<double> nodes;
    vector<double> weights;
    int degree;

    void initializeNodes()
    {
        nodes.resize(degree + 1);
        weights.resize(degree + 1);
        for (int i = 0; i <= degree; i++)
        {
            // 使用Chebyshev-Gauss节点
            nodes[i] = cos(M_PI * (2.0 * i + 1) / (2.0 * (degree + 1)));
            // 计算相应的权重
            weights[i] = M_PI / (degree + 1);
        }
    }

public:
    ChebyshevAccelerator(int deg = 16) : degree(deg)
    {
        initializeNodes();
    }

    VectorXcd accelerate(const function<VectorXcd(const VectorXcd &)> &matVecProduct,
                         const VectorXcd &vector,
                         const complex<double> &lower_bound,
                         const complex<double> &upper_bound)
    {
        VectorXcd result = VectorXcd::Zero(vector.size());

        // 计算映射参数
        complex<double> center = (upper_bound + lower_bound) * 0.5;
        complex<double> radius = (upper_bound - lower_bound) * 0.5;

        for (int i = 0; i <= degree; i++)
        {
            // 构造切比雪夫多项式
            complex<double> z = center + radius * complex<double>(nodes[i], 0);
            VectorXcd temp = vector;

            // 应用多项式
            for (int j = 0; j < 3; j++)
            { // 使用较低阶多项式以提高数值稳定性
                temp = matVecProduct(temp);
                temp *= complex<double>(1.0 / abs(z), 0); // 归一化以提高数值稳定性
            }

            result += weights[i] * temp;
        }

        return result / (degree + 1);
    }
};
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

class ImplicitArnoldiSolver
{
private:
    int n, m, max_restarts;
    const CustomMultiplier &multiplier;
    double tolerance;
    ChebyshevAccelerator accelerator;
    complex<double> computeRayleighQuotient(const VectorXcd &v)
    {
        VectorXcd Av = multiplier * v;
        return (v.adjoint() * Av)(0, 0) / (v.adjoint() * v)(0, 0);
    }
    void arnoldiIteration(MatrixXcd &V, MatrixXcd &H, int k_start, int k_end)
    {
        complex<double> lower_bound(0.5, -0.5); // 估计的特征值范围
        complex<double> upper_bound(1.5, 0.5);

        for (int j = k_start; j < k_end; j++)
        {
            // 使用切比雪夫加速
            VectorXcd w = accelerator.accelerate(
                [this](const VectorXcd &v)
                { return multiplier * v; },
                V.col(j),
                lower_bound,
                upper_bound);

            // 改进的正交化过程
            for (int i = 0; i <= j; i++)
            {
                complex<double> h = V.col(i).adjoint() * w;
                H(i, j) = h;
                w -= h * V.col(i);

                // 二次正交化
                complex<double> h2 = V.col(i).adjoint() * w;
                H(i, j) += h2;
                w -= h2 * V.col(i);
            }

            double norm_w = w.norm();
            if (norm_w < tolerance)
            {
                H.col(j).setZero();
                V.col(j + 1).setZero();
                break;
            }

            H(j + 1, j) = norm_w;
            V.col(j + 1) = w / norm_w;
        }
    }

    VectorXcd refineSingleVector(complex<double> &eigenvalue,
                                 const VectorXcd &initial_vector,
                                 int max_iter = 10)
    {
        VectorXcd current_vector = initial_vector;
        complex<double> prev_eigenvalue = eigenvalue;

        for (int iter = 0; iter < max_iter; ++iter)
        {
            VectorXcd new_vector = multiplier * current_vector;
            eigenvalue = computeRayleighQuotient(current_vector);

            new_vector.normalize();
            double diff = (new_vector - current_vector).norm();
            current_vector = new_vector;

            if (diff < tolerance && abs(eigenvalue - prev_eigenvalue) < tolerance)
            {
                break;
            }
            prev_eigenvalue = eigenvalue;
        }
        return current_vector;
    }

public:
    ImplicitArnoldiSolver(int size, int num_eigenvalues, const CustomMultiplier &mult,
                          int max_rest = 30, double tol = 1e-12)
        : n(size), m(num_eigenvalues * 2), max_restarts(max_rest),
          multiplier(mult), tolerance(tol), accelerator(16) {}

    void solve(vector<complex<double>> &eigenvalues, MatrixXcd &eigenvectors)
    {
        MatrixXcd V = MatrixXcd::Zero(n, m + 1);
        MatrixXcd H = MatrixXcd::Zero(m + 1, m);

        // 改进的初始向量选择
        VectorXcd initial = VectorXcd::Random(n);
        complex<double> lower_bound(0.5, -0.5);
        complex<double> upper_bound(1.5, 0.5);

        for (int i = 0; i < 3; ++i)
        {
            initial = accelerator.accelerate(
                [this](const VectorXcd &v)
                { return multiplier * v; },
                initial,
                lower_bound,
                upper_bound);
            initial.normalize();
        }
        V.col(0) = initial;

        // Main IRAM loop
        for (int restart = 0; restart < max_restarts; restart++)
        {
            arnoldiIteration(V, H, 0, m);

            MatrixXcd H_m = H.topLeftCorner(m, m);
            ComplexEigenSolver<MatrixXcd> ces(H_m);

            // Sort eigenvalues by magnitude
            vector<pair<double, int>> eig_order(m);
            for (int i = 0; i < m; i++)
            {
                eig_order[i] = make_pair(abs(ces.eigenvalues()(i)), i);
            }
            sort(eig_order.begin(), eig_order.end(), greater<pair<double, int>>());

            // Check convergence
            bool converged = true;
            for (int i = 0; i < m / 2; i++)
            {
                VectorXcd ritz_vector = V.leftCols(m) * ces.eigenvectors().col(eig_order[i].second);
                VectorXcd residual = multiplier * ritz_vector -
                                     ces.eigenvalues()(eig_order[i].second) * ritz_vector;
                if (residual.norm() > tolerance)
                {
                    converged = false;
                    break;
                }
            }

            if (converged)
                break;

            // Restart with best vectors
            MatrixXcd Q = ces.eigenvectors();
            V.leftCols(m) = V.leftCols(m) * Q;
            H.setZero();
        }

        // Final eigenvalue computation and refinement
        MatrixXcd H_m = H.topLeftCorner(m, m);
        ComplexEigenSolver<MatrixXcd> ces(H_m);

        eigenvalues.resize(m / 2);
        eigenvectors = MatrixXcd::Zero(n, m / 2);

        vector<pair<double, int>> eig_order(m);
        for (int i = 0; i < m; i++)
        {
            eig_order[i] = make_pair(abs(ces.eigenvalues()(i)), i);
        }
        sort(eig_order.begin(), eig_order.end(), greater<pair<double, int>>());

        for (int i = 0; i < m / 2; i++)
        {
            int idx = eig_order[i].second;
            complex<double> eigenvalue = ces.eigenvalues()(idx);
            VectorXcd eigenvector = V.leftCols(m) * ces.eigenvectors().col(idx);
            eigenvector.normalize();

            // Final refinement
            eigenvectors.col(i) = refineSingleVector(eigenvalue, eigenvector);
            eigenvalues[i] = eigenvalue;
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
        VectorXcd Av = multiplier * eigenvector;
        VectorXcd lambda_v = computed_eigenvalues[i] * eigenvector;

        double error = (Av - lambda_v).norm() / eigenvector.norm();
        errors[i] = make_pair(error, i);

        cout << "Eigenvalue " << i + 1 << ": " << computed_eigenvalues[i] << endl;
        cout << "Relative Error: " << error << endl;
        cout << "-------------------" << endl;
    }

    sort(errors.begin(), errors.end());
    cout << "\nBest 24 Results:" << endl;
    for (int i = 0; i < min(24, (int)errors.size()); ++i)
    {
        cout << "Eigenvalue " << errors[i].second + 1
             << " Relative Error: " << errors[i].first << endl;
    }
}

int main()
{
    const int matrix_size = 100;
    const int num_eigenvalues = 48;

    CustomMultiplier multiplier(matrix_size);
    ImplicitArnoldiSolver solver(matrix_size, num_eigenvalues, multiplier);

    vector<complex<double>> eigenvalues;
    MatrixXcd eigenvectors;
    solver.solve(eigenvalues, eigenvectors);

    verify_results(multiplier, eigenvalues, eigenvectors);

    return 0;
}