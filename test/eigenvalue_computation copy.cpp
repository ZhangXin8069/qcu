#include <iostream>
#include <iomanip>
#include <complex>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace Eigen;
using namespace std;
using Complex = std::complex<double>;
using ComplexMatrix = Matrix<Complex, Dynamic, Dynamic>;
using ComplexVector = Matrix<Complex, Dynamic, 1>;

// 进度条类
class ProgressBar {
private:
    std::mutex mutex_;
    int total_;
    int current_ = 0;
    int bar_width_ = 50;

public:
    ProgressBar(int total) : total_(total) {}

    void update(int progress) {
        std::lock_guard<std::mutex> lock(mutex_);
        current_ = progress;
        float percentage = static_cast<float>(current_) / total_;
        int filled = static_cast<int>(bar_width_ * percentage);

        std::cout << "\r[";
        for (int i = 0; i < bar_width_; ++i) {
            if (i < filled) 
                std::cout << "=";
            else 
                std::cout << " ";
        }
        std::cout << "] " << std::fixed << std::setprecision(1) 
                  << (percentage * 100.0) << "% " << current_ << "/" << total_;
        std::cout.flush();

        if (current_ >= total_) {
            std::cout << std::endl;
        }
    }
};

// 复数矩阵生成函数
class MatrixGenerator {
public:
    // 使用梅森旋转算法生成随机复数矩阵
    static ComplexMatrix generate(int rows, int cols, unsigned int seed = std::random_device{}()) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        ComplexMatrix matrix(rows, cols);
        
        ProgressBar progress(rows);
        
        #pragma omp parallel for
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                // 生成实部和虚部
                double real_part = dis(gen);
                double imag_part = dis(gen);
                matrix(i,j) = Complex(real_part, imag_part);
            }
            
            #pragma omp critical
            {
                progress.update(i + 1);
            }
        }

        return matrix;
    }
};

// Arnoldi迭代类
class ArnoldiIteration {
public:
    static pair<ComplexVector, ComplexMatrix> compute(
        const ComplexMatrix& A, 
        int k, 
        const ComplexVector& initial_vector
    ) {
        int n = A.rows();
        ComplexMatrix Q(n, k+1), H(k+1, k);
        Q.setZero();
        H.setZero();

        ProgressBar progress(k);

        // 归一化初始向量
        Q.col(0) = initial_vector / initial_vector.norm();

        for (int j = 0; j < k; ++j) {
            ComplexVector w = A * Q.col(j);

            for (int i = 0; i <= j; ++i) {
                H(i, j) = Q.col(i).adjoint() * w;
                w -= H(i, j) * Q.col(i);
            }

            H(j+1, j) = w.norm();

            // 防止除零
            if (std::abs(H(j+1, j)) > 1e-10 && j+1 < n) {
                Q.col(j+1) = w / H(j+1, j);
            }

            progress.update(j + 1);
        }

        // 提取部分Krylov子空间
        ComplexMatrix Q_reduced = Q.leftCols(k);
        ComplexMatrix H_reduced = H.topLeftCorner(k, k);

        // 计算约化Hessenberg矩阵的特征值
        SelfAdjointEigenSolver<ComplexMatrix> es(H_reduced.adjoint() * H_reduced);
        ComplexVector eigenvalues = es.eigenvalues().cast<Complex>();

        return {eigenvalues, Q_reduced};
    }
};

// 特征值验证函数
void validate_eigenvalues(
    const ComplexMatrix& A, 
    const ComplexVector& eigenvalues, 
    const ComplexMatrix& eigenvectors
) {
    cout << "\n--- Eigenvalue Validation ---" << endl;
    
    // 取第一个特征值和特征向量
    Complex first_eigenvalue = eigenvalues(0);
    ComplexVector first_eigenvector = eigenvectors.col(0);

    ComplexVector Av = A * first_eigenvector;
    ComplexVector lambda_v = first_eigenvalue * first_eigenvector;

    double norm_Av = Av.norm();
    double norm_lambda_v = lambda_v.norm();
    double error = (Av - lambda_v).norm() / norm_Av;

    cout << "First Eigenvalue: " << first_eigenvalue << endl;
    cout << "Eigenvector Norm: " << first_eigenvector.norm() << endl;
    cout << "Av Norm: " << norm_Av << endl;
    cout << "λv Norm: " << norm_lambda_v << endl;
    cout << "Relative Error: " << error << endl;
}

int main() {
    // 计算参数
    const int N = 100;  // 矩阵大小
    const int k = 60;    // Krylov子空间维度
    const int n = 50;    // 有效特征值数量

    // 打印基本信息
    cout << "Complex Eigenvalue Computation" << endl;
    cout << "Matrix Size: " << N << " x " << N << endl;
    cout << "Krylov Subspace Dimension: " << k << endl;

    // 生成复数矩阵
    cout << "\nGenerating Complex Matrix..." << endl;
    auto start = chrono::high_resolution_clock::now();
    
    ComplexMatrix A = MatrixGenerator::generate(N, N);
    
    auto matrix_end = chrono::high_resolution_clock::now();
    chrono::duration<double> matrix_time = matrix_end - start;
    cout << "Matrix Generation Time: " << matrix_time.count() << " seconds" << endl;

    // 生成随机初始向量
    ComplexVector v = ComplexVector::Random(N);

    // 方法1：直接特征值分解
    cout << "\n--- Method 1: Direct Eigenvalue Decomposition ---" << endl;
    start = chrono::high_resolution_clock::now();
    
    SelfAdjointEigenSolver<ComplexMatrix> es(A.adjoint() * A);
    ComplexVector direct_eigenvalues = es.eigenvalues().cast<Complex>();
    
    auto direct_end = chrono::high_resolution_clock::now();
    chrono::duration<double> direct_time = direct_end - start;
    cout << "Direct Eigenvalue Computation Time: " << direct_time.count() << " seconds" << endl;

    // 方法2：Arnoldi迭代
    cout << "\n--- Method 2: Arnoldi Iteration ---" << endl;
    start = chrono::high_resolution_clock::now();
    
    auto [arnoldi_eigenvalues, arnoldi_basis] = ArnoldiIteration::compute(A, k, v);
    
    auto arnoldi_end = chrono::high_resolution_clock::now();
    chrono::duration<double> arnoldi_time = arnoldi_end - start;
    cout << "Arnoldi Iteration Time: " << arnoldi_time.count() << " seconds" << endl;

    // 验证特征值
    cout << "\n=== Eigenvalue Validation ===" << endl;
    validate_eigenvalues(A, direct_eigenvalues, es.eigenvectors());
    validate_eigenvalues(A, arnoldi_eigenvalues, arnoldi_basis);

    return 0;
}