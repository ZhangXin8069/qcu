#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <unordered_set>
#include <random>

// 矩阵乘法（简单实现，可优化性能等）
std::vector<std::vector<double>> multiplyMatrices(const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B)
{
    int rowsA = A.size();
    int colsA = A[0].size();
    int colsB = B[0].size();
    std::vector<std::vector<double>> result(rowsA, std::vector<double>(colsB, 0.0));
    for (int i = 0; i < rowsA; ++i)
    {
        for (int j = 0; j < colsB; ++j)
        {
            for (int k = 0; k < colsA; ++k)
            {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// 矩阵转置（简单实现）
std::vector<std::vector<double>> transposeMatrix(const std::vector<std::vector<double>> &A)
{
    int rows = A.size();
    int cols = A[0].size();
    std::vector<std::vector<double>> result(cols, std::vector<double>(rows, 0.0));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            result[j][i] = A[i][j];
        }
    }
    return result;
}

// 计算两个向量的内积
double dotProduct(const std::vector<double> &v1, const std::vector<double> &v2)
{
    double result = 0.0;
    for (size_t i = 0; i < v1.size(); ++i)
    {
        result += v1[i] * v2[i];
    }
    return result;
}

// 向量减法，v1 - v2
std::vector<double> vectorSubtract(const std::vector<double> &v1, const std::vector<double> &v2)
{
    std::vector<double> result(v1.size(), 0.0);
    for (size_t i = 0; i < v1.size(); ++i)
    {
        result[i] = v1[i] - v2[i];
    }
    return result;
}

// 向量数乘，v1乘以标量alpha
std::vector<double> vectorScale(const std::vector<double> &v1, double alpha)
{
    std::vector<double> result(v1.size(), 0.0);
    for (size_t i = 0; i < v1.size(); ++i)
    {
        result[i] = v1[i] * alpha;
    }
    return result;
}

// 向量归一化
std::vector<double> normalizeVector(const std::vector<double> &v)
{
    double norm = std::sqrt(dotProduct(v, v));
    if (norm < 1e-16)
    { // 避免除以非常小的数，若范数过小则返回零向量
        return std::vector<double>(v.size(), 0.0);
    }
    return vectorScale(v, 1.0 / norm);
}

// 稀疏矩阵向量乘法，利用稀疏性只计算非零元素与对应向量元素的乘积
std::vector<double> sparseMatrixVectorMultiply(const std::vector<std::vector<std::pair<double, int>>> &A, const std::vector<double> &v)
{
    int rows = A.size();
    std::vector<double> result(rows, 0.0);
    for (int i = 0; i < rows; ++i)
    {
        for (const auto &nz_element : A[i])
        {
            result[i] += nz_element.first * v[nz_element.second];
        }
    }
    return result;
}

// Householder变换，用于QR分解，增加了数值稳定性判断
void householderTransformation(std::vector<std::vector<double>> &A, int k)
{
    int n = A.size();
    double alpha = 0.0;
    for (int i = k; i < n; ++i)
    {
        alpha += A[i][k] * A[i][k];
    }
    alpha = std::sqrt(alpha);
    if (A[k][k] < 0)
    {
        alpha = -alpha;
    }
    std::vector<double> v(n, 0.0);
    v[k] = A[k][k] - alpha;
    for (int i = k + 1; i < n; ++i)
    {
        v[i] = A[i][k];
    }
    double beta = 0.0;
    double vv = dotProduct(v, v);
    if (vv > 1e-16)
    { // 避免除以极小值导致数值问题
        beta = 2.0 / vv;
    }
    for (int j = k; j < n; ++j)
    {
        double s = 0.0;
        for (int i = k; i < n; ++i)
        {
            s += A[i][k] * A[i][j];
        }
        s *= beta;
        for (int i = k; i < n; ++i)
        {
            A[i][j] -= s * v[i];
        }
    }
    for (int i = 0; i < n; ++i)
    {
        double s = 0.0;
        for (int j = k; j < n; ++j)
        {
            s += A[i][j] * v[j];
        }
        s *= beta;
        for (int j = k; j < n; ++j)
        {
            A[i][j] -= s * v[j];
        }
    }
    A[k][k] -= alpha;
}

// QR分解，用于将Hessenberg矩阵分解为正交矩阵Q和上三角矩阵R
void qrDecomposition(const std::vector<std::vector<double>> &H, std::vector<std::vector<double>> &Q, std::vector<std::vector<double>> &R)
{
    int m = H.size();
    Q = std::vector<std::vector<double>>(m, std::vector<double>(m, 0.0));
    R = std::vector<std::vector<double>>(m, std::vector<double>(m, 0.0));
    std::vector<std::vector<double>> H_copy = H;
    for (int k = 0; k < m - 1; ++k)
    {
        householderTransformation(H_copy, k);
    }
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            if (i <= j)
            {
                R[i][j] = H_copy[i][j];
            }
            if (i >= j)
            {
                Q[i][j] = H_copy[i][j];
            }
        }
    }
}

// 利用QR迭代法求解Hessenberg矩阵的特征值，增加收敛判断改进并设置最大迭代次数上限，同时优化收敛判断逻辑
std::vector<double> qrIteration(const std::vector<std::vector<double>> &H, int numEigs)
{
    int m = H.size();
    std::vector<std::vector<double>> Q, R;
    std::vector<std::vector<double>> H_copy = H;
    const int maxIterations = 1000; // 适当增加最大迭代次数，同时设置一个相对合理的上限避免无限循环
    const double tolerance = 1e-12; // 调整收敛容差
    bool converged = false;
    int iter_count = 0;
    std::vector<double> prev_eigenvalues(m, 0.0);
    while (iter_count < maxIterations && !converged)
    {
        qrDecomposition(H_copy, Q, R);
        std::vector<std::vector<double>> new_H(m, std::vector<double>(m, 0.0));
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                for (int k = 0; k < m; ++k)
                {
                    new_H[i][j] += R[i][k] * Q[k][j];
                }
            }
        }
        converged = true;
        // 优化收敛判断逻辑，比较相邻两次迭代得到的特征值向量（取对角元素）
        for (int i = 0; i < m; ++i)
        {
            if (std::abs(prev_eigenvalues[i] - new_H[i][i]) > tolerance)
            {
                converged = false;
                break;
            }
        }
        prev_eigenvalues = std::vector<double>(new_H.size(), 0.0);
        for (int i = 0; i < m; ++i)
        {
            prev_eigenvalues[i] = new_H[i][i];
        }
        H_copy = new_H;
        iter_count++;
    }
    std::vector<double> eigenvalues(m);
    for (int i = 0; i < m; ++i)
    {
        eigenvalues[i] = H_copy[i][i];
    }
    return eigenvalues;
}

// Arnoldi方法实现，适配稀疏矩阵乘法，返回正交基矩阵V
std::pair<std::vector<double>, std::vector<std::vector<double>>> arnoldiMethod(const std::vector<std::vector<std::pair<double, int>>> &A, int m, int numEigs)
{
    int n = A.size();
    std::vector<std::vector<double>> V(n, std::vector<double>(m, 0.0)); // 正交基矩阵
    std::vector<std::vector<double>> H(m, std::vector<double>(m, 0.0)); // Hessenberg矩阵

    std::vector<double> v(n, 0.0);
    srand(static_cast<unsigned int>(time(nullptr))); // 设置随机种子，使每次运行随机情况不同
    for (int i = 0; i < n; ++i)
    {
        v[i] = static_cast<double>(rand()) / RAND_MAX; // 生成随机初始向量
    }
    v = normalizeVector(v);

    for (int j = 0; j < m; ++j)
    {
        std::vector<double> w = sparseMatrixVectorMultiply(A, V[j]);
        for (int i = 0; i <= j; ++i)
        {
            H[i][j] = dotProduct(V[i], w);
            w = vectorSubtract(w, vectorScale(V[i], H[i][j]));
        }
        double h_jp1j = std::sqrt(dotProduct(w, w));
        if (h_jp1j < 1e-16)
        { // 避免数值问题，若新向量范数过小则停止迭代
            break;
        }
        H[j + 1][j] = h_jp1j;
        for (int k = 0; k < n; ++k)
        {
            V[k][j + 1] = w[k] / h_jp1j;
        }
    }

    // 使用QR迭代法求解Hessenberg矩阵H的特征值
    std::vector<double> eigenvalues = qrIteration(H, numEigs);
    return std::make_pair(eigenvalues, V);
}

// 生成随机正交矩阵（通过Householder变换来构造，简单示例，可优化）
std::vector<std::vector<double>> generateRandomOrthogonalMatrix(int size)
{
    std::vector<std::vector<double>> Q(size, std::vector<double>(size, 0.0));
    std::vector<std::vector<double>> I(size, std::vector<double>(size, 0.0));
    for (int i = 0; i < size; ++i)
    {
        I[i][i] = 1.0;
    }
    std::vector<std::vector<double>> A = I;
    for (int k = 0; k < size - 1; ++k)
    {
        householderTransformation(A, k);
    }
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            Q[i][j] = A[i][j];
        }
    }
    return Q;
}

// 生成随机的大特征向量（这里向量维度设为100，元素取值范围在 -10到10之间，可按需调整）
std::vector<std::vector<double>> generateRandomLargeEigenvectors(int numEigenvectors, int vectorDimension)
{
    std::vector<std::vector<double>> eigenvectors(numEigenvectors, std::vector<double>(vectorDimension, 0.0));
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-10.0, 10.0);
    for (int i = 0; i < numEigenvectors; ++i)
    {
        for (int j = 0; j < vectorDimension; ++j)
        {
            eigenvectors[i][j] = distribution(generator);
        }
    }
    return eigenvectors;
}

// 根据已知特征值和特征向量生成大规模稀疏矩阵（简单示例，通过相似变换模拟稀疏性）
std::vector<std::vector<std::pair<double, int>>> generateSparseMatrixFromEigen(const std::vector<double> &eigenvalues, const std::vector<std::vector<double>> &eigenvectors, double sparsity)
{
    int n = eigenvalues.size();
    std::vector<std::vector<double>> diagMatrix(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
    {
        diagMatrix[i][i] = eigenvalues[i];
    }
    std::vector<std::vector<double>> Q = generateRandomOrthogonalMatrix(n);
    std::vector<std::vector<double>> A = multiplyMatrices(multiplyMatrices(Q, diagMatrix), transposeMatrix(Q));
    std::vector<std::vector<std::pair<double, int>>> sparseMatrix(n, std::vector<std::pair<double, int>>());
    int numNonZeros = static_cast<int>(n * n * sparsity);
    std::unordered_set<int> rowIndices;
    std::unordered_set<int> colIndices;
    while (rowIndices.size() < numNonZeros)
    {
        rowIndices.insert(rand() % n);
    }
    while (colIndices.size() < numNonZeros)
    {
        colIndices.insert(rand() % n);
    }
    std::vector<int> rowVec(rowIndices.begin(), rowIndices.end());
    std::vector<int> colVec(colIndices.begin(), colIndices.end());
    std::random_shuffle(rowVec.begin(), rowVec.end());
    std::random_shuffle(colVec.begin(), colVec.end());
    for (int i = 0; i < numNonZeros; ++i)
    {
        sparseMatrix[rowVec[i]].push_back(std::make_pair(A[rowVec[i]][colVec[i]], colVec[i]));
    }
    return sparseMatrix;
}

// 验证特征值和特征向量关系的函数，通过计算 Av - λv 的范数来判断是否近似满足等式
bool verifyEigenvalues(const std::vector<std::vector<std::pair<double, int>>> &A, const std::vector<double> &eigenvalues, const std::vector<std::vector<double>> &V, double tolerance)
{
    int numEigs = eigenvalues.size();
    int n = A.size();
    for (int i = 0; i < numEigs; ++i)
    {
        std::vector<double> v = V[n - 1]; // 简单取 V 的最后一列作为近似特征向量（只是一种简单方式）
        std::vector<double> Av = sparseMatrixVectorMultiply(A, v);
        std::vector<double> lambda_v = vectorScale(v, eigenvalues[i]);
        std::vector<double> diff = vectorSubtract(Av, lambda_v);
        double norm_diff = std::sqrt(dotProduct(diff, diff));
        if (norm_diff > tolerance)
        {
            return false;
        }
    }
    return true;
}
int main()
{
    try
    {
        // 已知的特征值（简单示例，可按需调整）
        std::vector<double> knownEigenvalues = {1.0, 2.0, 3.0};

        // 生成随机的大特征向量，这里设为生成3个特征向量，每个向量维度为100（可按需调整）
        std::vector<std::vector<double>> knownEigenvectors = generateRandomLargeEigenvectors(3, 100);

        double sparsity = 0.1; // 稀疏度，可根据实际情况调整，这里表示10%的元素为非零元素

        // 根据已知特征值和特征向量生成稀疏矩阵
        std::vector<std::vector<std::pair<double, int>>> sparseMatrix = generateSparseMatrixFromEigen(knownEigenvalues, knownEigenvectors, sparsity);

        int krylovDim = 20;       // Krylov子空间维度，可根据情况调整
        int numEigsToCompute = 3; // 要计算的特征值数量，与已知特征值数量保持一致

        // 使用Arnoldi方法求解特征值与特征向量
        std::pair<std::vector<double>, std::vector<std::vector<double>>> result = arnoldiMethod(sparseMatrix, krylovDim, numEigsToCompute);
        std::vector<double> computedEigenvalues = result.first;
        std::vector<std::vector<double>> computedEigenvectors = result.second;

        std::cout << "Approximate eigenvalues: ";
        for (double eig : computedEigenvalues)
        {
            std::cout << std::fixed << std::setprecision(6) << eig << " ";
        }
        std::cout << std::endl;

        // 进行验证，设置合适的误差容忍度，这里设为1e-6，可根据实际情况调整
        bool isVerified = verifyEigenvalues(sparseMatrix, computedEigenvalues, computedEigenvectors, 1e-6);
        if (isVerified)
        {
            std::cout << "The calculated eigenvalues and eigenvectors are verified within the tolerance." << std::endl;
        }
        else
        {
            std::cout << "The verification of the calculated eigenvalues and eigenvectors failed." << std::endl;
        }
    }
    catch (...)
    {
        std::cerr << "An error occurred during the calculation." << std::endl;
        return -1;
    }
    return 0;
}
