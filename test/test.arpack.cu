#include <iostream>
#include <vector>
#include <cmath>
#include <arpack++.h>

int main() {
    // 定义矩阵的大小
    int n = 5; // 矩阵的维度
    int nev = 2; // 需要计算的特征值个数

    // 使用ARPACK创建一个EigenSolver对象，选择单精度
    ARPACK::SymStdEig<float> solver(n, nev);

    // 定义矩阵 (这里以一个对称矩阵为例)
    std::vector<float> matrix = {
        4.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 4.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 4.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 4.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 1.0f, 4.0f
    };

    // 设置矩阵的大小和初始猜测
    solver.setMatrix(matrix.data());

    // 调用ARPACK计算特征值
    solver.solve();

    // 输出结果
    std::cout << "Eigenvalues (single precision): " << std::endl;
    for (int i = 0; i < nev; ++i) {
        std::cout << "λ" << i + 1 << ": " << solver.eigenvalue(i) << std::endl;
    }

    return 0;
}
