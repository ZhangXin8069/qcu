#include <iostream>
#include <complex>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include "arpackf.h"

using namespace std;
using Complex = complex<double>;
using Clock = chrono::high_resolution_clock;

// 生成随机复数矩阵
vector<Complex> generate_random_matrix(int n)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1.0, 1.0);

    vector<Complex> matrix(n * n);
    for (int i = 0; i < n * n; i++)
    {
        matrix[i] = Complex(dis(gen), dis(gen));
    }
    return matrix;
}

// 矩阵向量乘法
void matrix_vector_product(const vector<Complex> &matrix, Complex *x, Complex *y, int n)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = Complex(0.0, 0.0);
        for (int j = 0; j < n; j++)
        {
            y[i] += matrix[i * n + j] * x[j];
        }
    }
}

// 验证特征值和特征向量
double verify_eigenpairs(const vector<Complex> &matrix, const vector<Complex> &eigenvalues,
                         const vector<Complex> &eigenvectors, int n, int nev)
{
    double max_error = 0.0;
    vector<Complex> Ax(n);
    vector<Complex> lambda_x(n);

    for (int i = 0; i < nev; i++)
    {
        // 计算 Ax
        matrix_vector_product(matrix, &eigenvectors[i * n], Ax.data(), n);

        // 计算 λx
        for (int j = 0; j < n; j++)
        {
            lambda_x[j] = eigenvalues[i] * eigenvectors[i * n + j];
        }

        // 计算 ||Ax - λx||/||λx||
        double error = 0.0;
        double norm = 0.0;
        for (int j = 0; j < n; j++)
        {
            error += norm(Ax[j] - lambda_x[j]);
            norm += norm(lambda_x[j]);
        }
        error = sqrt(error / norm);
        max_error = max(max_error, error);
    }
    return max_error;
}

int main()
{
    int n = 100;           // 矩阵维度
    int nev = 5;           // 要计算的特征值数量
    int ncv = 2 * nev + 1; // Arnoldi向量数量

    // 生成随机矩阵
    cout << "生成随机复数矩阵..." << endl;
    vector<Complex> matrix = generate_random_matrix(n);

    // ARPACK参数设置
    char bmat[2] = "I";   // 标准特征值问题
    char which[3] = "LM"; // 计算最大模特征值
    int ido = 0;
    int info = 0;
    double tol = 1e-10;
    int lworkl = ncv * (3 * ncv + 5);

    vector<int> iparam(11, 0);
    iparam[0] = 1;    // ishift
    iparam[2] = 3000; // 最大迭代次数
    iparam[3] = 1;    // NB
    iparam[6] = 1;    // 模式1

    // 分配内存
    vector<Complex> resid(n);
    vector<Complex> v(n * ncv);
    vector<Complex> workd(3 * n);
    vector<Complex> workl(lworkl);
    vector<int> ipntr(14);
    vector<Complex> d(nev);
    vector<Complex> z(n * nev);
    vector<int> select(ncv);
    vector<double> rwork(ncv);

    // 开始计时
    auto start = Clock::now();

    // ARPACK迭代
    while (ido != 99)
    {
        znaupd(&ido, bmat, &n, which, &nev, &tol, resid.data(),
               &ncv, v.data(), &n,
               iparam.data(), ipntr.data(), workd.data(),
               workl.data(), &lworkl, rwork.data(), &info,
               1, 2);

        if (ido == -1 || ido == 1)
        {
            matrix_vector_product(matrix,
                                  &workd[ipntr[0] - 1],
                                  &workd[ipntr[1] - 1],
                                  n);
        }
    }

    if (info < 0)
    {
        cout << "Error in znaupd: " << info << endl;
        return 1;
    }

    // 计算特征值和特征向量
    int rvec = 1;
    char howmny[2] = "A";
    Complex sigma(0.0, 0.0);

    zneupd(&rvec, howmny, select.data(),
           d.data(), z.data(), &n,
           &sigma, bmat, &n, which,
           &nev, &tol, resid.data(),
           &ncv, v.data(), &n,
           iparam.data(), ipntr.data(),
           workd.data(), workl.data(),
           &lworkl, rwork.data(), &info,
           1, 1, 2);

    // 结束计时
    auto end = Clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    // 验证结果
    double error = verify_eigenpairs(matrix, d, z, n, nev);

    // 输出结果
    cout << "\n计算结果：" << endl;
    cout << "矩阵大小: " << n << " x " << n << endl;
    cout << "计算用时: " << duration.count() << " 毫秒" << endl;
    cout << "最大相对误差: " << error << endl;
    cout << "\n特征值：" << endl;
    for (int i = 0; i < nev; i++)
    {
        cout << "λ" << i + 1 << " = " << d[i]
             << " (|λ| = " << abs(d[i]) << ")" << endl;
    }

    cout << "\n迭代信息：" << endl;
    cout << "执行的迭代次数: " << iparam[2] << endl;
    cout << "矩阵操作次数: " << iparam[8] << endl;

    return 0;
}