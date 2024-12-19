#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>

using namespace Eigen;
using namespace std;

// 计算Chebyshev多项式的值
double chebyshev_polynomial(int k, double theta) {
    if (k == 0) return 1.0;
    if (k == 1) return theta;
    double T0 = 1.0, T1 = theta, T2;
    for (int i = 2; i <= k; ++i) {
        T2 = 2.0 * theta * T1 - T0;
        T0 = T1;
        T1 = T2;
    }
    return T1;
}

// 幂法估算最大特征值
double power_method_max_eigenvalue(const SparseMatrix<double>& A, int max_iter = 1000, double tol = 1e-6) {
    int n = A.rows();
    VectorXd b = VectorXd::Random(n);
    b.normalize();

    double eigenvalue = 0.0;
    for (int i = 0; i < max_iter; ++i) {
        VectorXd Ab = A * b;
        double new_eigenvalue = b.dot(Ab);
        if (fabs(new_eigenvalue - eigenvalue) < tol) {
            break;
        }
        eigenvalue = new_eigenvalue;
        b = Ab.normalized();
    }

    return eigenvalue;
}

// 反幂法估算最小特征值
double power_method_min_eigenvalue(const SparseMatrix<double>& A, int max_iter = 1000, double tol = 1e-6) {
    int n = A.rows();
    VectorXd b = VectorXd::Random(n);
    b.normalize();

    double eigenvalue = 0.0;
    for (int i = 0; i < max_iter; ++i) {
        VectorXd Ab = A * b;
        double new_eigenvalue = b.dot(Ab);
        if (fabs(new_eigenvalue - eigenvalue) < tol) {
            break;
        }
        eigenvalue = new_eigenvalue;
        b = Ab.normalized();
    }

    return 1.0 / eigenvalue;  // 反幂法返回最小特征值
}

// BICGSTAB 算法实现
bool bicgstab(const SparseMatrix<double>& A, const VectorXd& b, VectorXd& x, int maxIter = 1000, double tol = 1e-6) {
    int n = A.rows();
    VectorXd r = b - A * x;  // 初始残差
    VectorXd r_hat = r;      // r_hat是r的初始副本
    VectorXd p = r;          // p是搜索方向
    VectorXd v = VectorXd::Zero(n);
    VectorXd s = VectorXd::Zero(n);
    VectorXd t = VectorXd::Zero(n);

    double alpha = 1.0;
    double beta = 0.0;
    double omega = 1.0;
    double rho = r_hat.dot(r); // 初始rho值

    double rho_old = 1.0; // 初始化rho_old

    for (int iter = 0; iter < maxIter; ++iter) {
        rho = r_hat.dot(r);
        if (rho == 0) {
            cout << "BICGSTAB failure: rho is zero." << endl;
            return false;
        }

        if (iter == 0) {
            p = r;
        } else {
            beta = (rho / rho_old) * (alpha / omega);
            p = r + beta * (p - omega * v);
        }

        v = A * p;
        alpha = rho / r_hat.dot(v);

        s = r - alpha * v;
        if (s.norm() < tol) {
            x += alpha * p;
            cout << "Converged in " << iter << " iterations." << endl;
            return true;
        }

        t = A * s;
        omega = t.dot(s) / t.dot(t);

        x += alpha * p + omega * s;
        r = s - omega * t;

        rho_old = rho;  // 更新rho_old
    }

    cout << "BICGSTAB did not converge." << endl;
    return false;
}

// Chebyshev优化BICGSTAB
bool bicgstab_chebyshev(const SparseMatrix<double>& A, const VectorXd& b, VectorXd& x, int maxIter = 1000, double tol = 1e-6, int m = 2) {
    int n = A.rows();
    VectorXd r = b - A * x;
    VectorXd r_hat = r;
    VectorXd p = r;
    VectorXd v = VectorXd::Zero(n);
    VectorXd s = VectorXd::Zero(n);
    VectorXd t = VectorXd::Zero(n);

    double alpha = 1.0;
    double beta = 0.0;
    double omega = 1.0;
    double rho_old = 1.0;

    // 使用幂法估算最大特征值和最小特征值
    double lambda_max = power_method_max_eigenvalue(A);
    double lambda_min = power_method_min_eigenvalue(A);
    double theta_0 = 2.0 / (lambda_max - lambda_min);  // 使用更加合适的特征值范围估算

    for (int iter = 0; iter < maxIter; ++iter) {
        double rho = r_hat.dot(r);
        if (rho == 0) {
            cout << "BICGSTAB failure: rho is zero." << endl;
            return false;
        }

        if (iter == 0) {
            p = r;
        } else {
            beta = (rho / rho_old) * (alpha / omega);
            p = r + beta * (p - omega * v);
        }

        v = A * p;

        // 应用Chebyshev预处理
        double theta = theta_0 * (v.norm() / r.norm());
        for (int k = 0; k < m; ++k) {
            double T = chebyshev_polynomial(k, theta);
            v *= T;
        }

        alpha = rho / r_hat.dot(v);
        s = r - alpha * v;
        if (s.norm() < tol) {
            x += alpha * p;
            cout << "Converged in " << iter << " iterations." << endl;
            return true;
        }

        t = A * s;
        omega = t.dot(s) / t.dot(t);

        x += alpha * p + omega * s;
        r = s - omega * t;

        rho_old = rho;
    }

    cout << "BICGSTAB did not converge." << endl;
    return false;
}

int main() {
    // 创建一个大矩阵A和一个随机的向量b
    const int N = 10000;  // 设置矩阵的大小
    SparseMatrix<double> A(N, N);
    VectorXd b = VectorXd::Random(N);
    VectorXd x = VectorXd::Zero(N);

    // 构造一个带有一定结构的稀疏矩阵
    for (int i = 0; i < N; ++i) {
        A.insert(i, i) = 4.0;  // 对角线元素
        if (i > 0) A.insert(i, i - 1) = -1.0;  // 上三角带元素
        if (i < N - 1) A.insert(i, i + 1) = -1.0;  // 下三角带元素
    }

    // BICGSTAB算法优化前计时
    auto start = chrono::high_resolution_clock::now();
    bicgstab(A, b, x);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "BICGSTAB (no Chebyshev) took: " << duration.count() << " ms" << endl;

    // 重置初始解
    x = VectorXd::Zero(N);

    // BICGSTAB算法优化后计时
    start = chrono::high_resolution_clock::now();
    bicgstab_chebyshev(A, b, x, 1000, 1e-6, 1000);  // 尝试更小的m值
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "BICGSTAB (with Chebyshev) took: " << duration.count() << " ms" << endl;

    return 0;
}
