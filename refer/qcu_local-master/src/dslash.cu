#include <cstdio>
#include <cmath>
#include <assert.h>
#include <chrono>

#include "../../include/qcu.h"
// #define ALTER
// #define DEBUG
#define FRONT 1
#define BACK 0

#define Nc 3
#define Nd 4
#define Ns 4
#define N_ 12
#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define checkCudaErrors(err)                                                                                          \
  {                                                                                                                   \
    if (err != cudaSuccess) {                                                                                         \
      fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", err,                    \
              cudaGetErrorString(err), __FILE__, __LINE__);                                                           \
      exit(-1);                                                                                                       \
    }                                                                                                                 \
  }

class Complex {
private:
  double real_;
  double imag_;

public:
  __device__ __host__ Complex(double real, double imag) : real_(real), imag_(imag) {}
  Complex() = default;
  __device__ __host__ Complex(const Complex &complex) : real_(complex.real_), imag_(complex.imag_) {}
  __device__ __host__ double norm2() { return sqrt(real_ * real_ + imag_ * imag_); }
  __device__ __host__ void setImag(double imag) { imag_ = imag; }
  __device__ __host__ void setReal(double real) { real_ = real; }
  __device__ __host__ double real() const { return real_; }
  __device__ __host__ double imag() const { return imag_; }

  __device__ __host__ Complex &operator=(const Complex &complex)
  {
    real_ = complex.real_;
    imag_ = complex.imag_;
    return *this;
  }
  __device__ __host__ Complex &operator=(double rhs)
  {
    real_ = rhs;
    imag_ = 0;
    return *this;
  }
  __device__ __host__ Complex operator+(const Complex &complex) const
  {
    return Complex(real_ + complex.real_, imag_ + complex.imag_);
  }
  __device__ __host__ Complex operator-(const Complex &complex) const
  {
    return Complex(real_ - complex.real_, imag_ - complex.imag_);
  }
  __device__ __host__ Complex operator-() const { return Complex(-real_, -imag_); }
  __device__ __host__ Complex operator*(const Complex &rhs) const
  {
    return Complex(real_ * rhs.real_ - imag_ * rhs.imag_, real_ * rhs.imag_ + imag_ * rhs.real_);
  }
  __device__ __host__ Complex operator*(const double &rhs) const { return Complex(real_ * rhs, imag_ * rhs); }
  __device__ __host__ Complex &operator*=(const Complex &rhs)
  {
    real_ = real_ * rhs.real_ - imag_ * rhs.imag_;
    imag_ = real_ * rhs.imag_ + imag_ * rhs.real_;
    return *this;
  }
  __device__ __host__ Complex &operator*=(const double &rhs)
  {
    real_ = real_ * rhs;
    imag_ = imag_ * rhs;
    return *this;
  }
  __device__ __host__ Complex operator/(const double &rhs) { return Complex(real_ / rhs, imag_ / rhs); }
  __device__ __host__ Complex operator/(const Complex &rhs) const
  {
    return (*this * rhs.conj()) / (rhs.real() * rhs.real() + rhs.imag() * rhs.imag());
  }
  __device__ __host__ Complex &operator/=(const Complex &rhs)
  {
    double new_real = (real_ * rhs.real() + imag_ * rhs.imag()) / (rhs.real() * rhs.real() + rhs.imag() * rhs.imag());
    double new_imag = (rhs.real() * imag_ - real_ * rhs.imag()) / (rhs.real() * rhs.real() + rhs.imag() * rhs.imag());
    real_ = new_real;
    imag_ = new_imag;
    return *this;
  }
  __device__ __host__ Complex &operator+=(const Complex &rhs)
  {
    real_ += rhs.real_;
    imag_ += rhs.imag_;
    return *this;
  }

  __device__ __host__ Complex &operator-=(const Complex &rhs)
  {
    real_ -= rhs.real_;
    imag_ -= rhs.imag_;
    return *this;
  }

  __device__ __host__ Complex &clear2Zero()
  {
    real_ = 0;
    imag_ = 0;
    return *this;
  }
  __device__ __host__ Complex conj() const { return Complex(real_, -imag_); }
  __device__ __host__ bool operator==(const Complex &rhs) { return real_ == rhs.real_ && imag_ == rhs.imag_; }
  __device__ __host__ bool operator!=(const Complex &rhs) { return real_ != rhs.real_ || imag_ != rhs.imag_; }
  __device__ __host__ void output() const { printf("(%lf + %lfi)", real_, imag_); }
};

// FROM newbing

// 定义一个常量，表示矩阵的大小
#define N 12

// 定义一个函数，用于交换两行
__device__ void swapRows(Complex *matrix, int row1, int row2)
{
  Complex temp;
  for (int i = 0; i < N * 2; i++) {
    temp = matrix[row1 * N * 2 + i];
    matrix[row1 * N * 2 + i] = matrix[row2 * N * 2 + i];
    matrix[row2 * N * 2 + i] = temp;
  }
}
// 定义一个函数，用于将一行除以一个数
__device__ void divideRow(Complex *matrix, int row, Complex num)
{
  for (int i = 0; i < N * 2; i++) {
    matrix[row * N * 2 + i] /= num;
  }
}
// 定义一个函数，用于将一行减去另一行乘以一个数
__device__ void subtractRow(Complex *matrix, int row1, int row2, Complex num)
{
  for (int i = 0; i < N * 2; i++) {
    matrix[row1 * N * 2 + i] -= num * matrix[row2 * N * 2 + i];
  }
}
// 定义一个函数，用于打印矩阵
__device__ void printMatrix(Complex *matrix)
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N * 2; j++) {
      matrix[i * N * 2 + j].output();
    }
    printf("\n");
  }
}
// 定义一个函数，用于求逆矩阵
__device__ void inverseMatrix(Complex *matrix, Complex *result)
{
  // 创建一个单位矩阵
  // double* identity = (double*)malloc(sizeof(double) * N * N);
  Complex identity[N * N];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (i == j) {
        identity[i * N + j] = Complex(1, 0);
      } else {
        identity[i * N + j].clear2Zero();
      }
    }
  }
  // 将原矩阵和单位矩阵平接在一起，形成增广矩阵
  Complex augmented[N * N * 2];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N * 2; j++) {
      if (j < N) {
        augmented[i * N * 2 + j] = matrix[i * N + j];
      } else {
        augmented[i * N * 2 + j] = identity[i * N + (j - N)];
      }
    }
  }

  // 对增广矩阵进行高斯消元法
  for (int i = 0; i < N; i++) {
    // 如果对角线上的元素为0，就交换两行
    if (augmented[i * N * 2 + i] == Complex(0, 0)) {
      for (int j = i + 1; j < N; j++) {
        if (augmented[j * N * 2 + i] != Complex(0, 0)) {
          swapRows(augmented, i, j);
          break;
        }
      }
    }

    // 如果对角线上的元素不为1，就将该行除以该元素
    if (augmented[i * N * 2 + i] != Complex(1, 0)) {
      divideRow(augmented, i, augmented[i * N * 2 + i]);
    }

    // 将其他行减去该行乘以相应的系数，使得该列除了对角线上的元素外都为0
    for (int j = 0; j < N; j++) {
      if (j != i) {
        subtractRow(augmented, j, i, augmented[j * N * 2 + i]);
      }
    }
  }

  // 从增广矩阵中分离出逆矩阵
  // double* inverse = (double*)malloc(sizeof(double) * N * N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      // inverse[i * N + j] = augmented[i * N * 2 + (j + N)];
      result[i * N + j] = augmented[i * N * 2 + (j + N)];
    }
  }
}

// end (FROM  newbing)

//--------------------------------------------
class Point {
private:
  int x_;
  int y_;
  int z_;
  int t_;
  int parity_;

public:
  Point() = default;
  __device__ __host__ Point(const Point &rhs) : x_(rhs.x_), y_(rhs.y_), z_(rhs.z_), t_(rhs.t_), parity_(rhs.parity_) {}
  __device__ __host__ Point(int x, int y, int z, int t, int parity) : x_(x), y_(y), z_(z), t_(t), parity_(parity) {}
  __device__ __host__ void outputInfo()
  {
    printf("Point: (x,y,z,t)=(%d, %d, %d, %d), parity = %d\n", x_, y_, z_, t_, parity_);
  }
  __device__ __host__ int getParity() const { return parity_; }
  __device__ __host__ Point move(int front_back, int direction, int Lx, int Ly, int Lz, int Lt) const
  { // direction +-1234
    // 1-front 0-back
    assert(abs(direction) >= 0 && abs(direction) < 4);
    assert(front_back == BACK || front_back == FRONT);

    int new_pos;
    int eo = (y_ + z_ + t_) & 0x01; // (y+z+t)%2

    if (direction == 0) {
      if (!front_back) {
        new_pos = x_ + (eo == parity_) * (-1 + (x_ == 0) * Lx);
        return Point(new_pos, y_, z_, t_, 1 - parity_);
      } else {
        new_pos = x_ + (eo != parity_) * (1 + (x_ == Lx - 1) * (-Lx));
        return Point(new_pos, y_, z_, t_, 1 - parity_);
      }
    } else if (direction == 1) { // y 前进
      if (!front_back) {
        new_pos = y_ - 1 + (y_ == 0) * Ly;
        return Point(x_, new_pos, z_, t_, 1 - parity_);
      } else {
        new_pos = y_ + 1 + (y_ == Ly - 1) * (-Ly);
        return Point(x_, new_pos, z_, t_, 1 - parity_);
      }
    } else if (direction == 2) {
      if (!front_back) {
        new_pos = z_ - 1 + (z_ == 0) * Lz;
        return Point(x_, y_, new_pos, t_, 1 - parity_);
      } else {
        new_pos = z_ + 1 + (z_ == Lz - 1) * (-Lz);
        return Point(x_, y_, new_pos, t_, 1 - parity_);
      }
    } else if (direction == 3) {
      if (!front_back) {
        new_pos = t_ - 1 + (t_ == 0) * Lt;
        return Point(x_, y_, z_, new_pos, 1 - parity_);
      } else {
        new_pos = t_ + 1 + (t_ == Lt - 1) * (-Lt);
        return Point(x_, y_, z_, new_pos, 1 - parity_);
      }
    } else {
      return *this;
    }
  }

  __device__ __host__ Complex *getPointGauge(Complex *origin, int direction, int Lx, int Ly, int Lz, int Lt) const
  {
    return origin + (((((((direction << 1) + parity_) * Lt + t_) * Lz + z_) * Ly + y_) * Lx) + x_) * Nc * Nc;
  }

  __device__ __host__ Complex *getPointVector(Complex *origin, int Lx, int Ly, int Lz, int Lt) const
  {
    return origin + (((t_ * Lz + z_) * Ly + y_) * Lx + x_) * Ns * Nc;
  }
  __device__ __host__ Point &operator=(const Point &rhs)
  {
    x_ = rhs.x_;
    y_ = rhs.y_;
    z_ = rhs.z_;
    t_ = rhs.t_;
    parity_ = rhs.parity_;
    return *this;
  }
  __device__ __host__ Complex *getPointClover(Complex *origin, int Lx, int Ly, int Lz, int Lt) const
  {
    return origin + (((((parity_ * Lt + t_) * Lz + z_) * Ly + y_) * Lx) + x_) * Nc * Ns * Nc * Ns;
  }
};

__device__ __host__ inline void reconstructSU3(Complex *su3)
{
  su3[6] = (su3[1] * su3[5] - su3[2] * su3[4]).conj();
  su3[7] = (su3[2] * su3[3] - su3[0] * su3[5]).conj();
  su3[8] = (su3[0] * su3[4] - su3[1] * su3[3]).conj();
}

__device__ __host__ inline void loadGauge(Complex *u_local, void *gauge_ptr, int direction, const Point &p, int Lx,
                                          int Ly, int Lz, int Lt)
{
  Complex *u = p.getPointGauge(static_cast<Complex *>(gauge_ptr), direction, Lx, Ly, Lz, Lt);
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    u_local[i] = u[i];
  }
  reconstructSU3(u_local);
}
__device__ __host__ inline void loadVector(Complex *src_local, void *fermion_in, const Point &p, int Lx, int Ly,
                                           int Lz, int Lt)
{
  Complex *src = p.getPointVector(static_cast<Complex *>(fermion_in), Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src[i];
  }
}

__device__ __host__ void gaugeMul(Complex *result, Complex *u1, Complex *u2)
{
  Complex temp;
  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      temp.clear2Zero();
      for (int k = 0; k < Nc; k++) {
        // result[i * Nc + j] = u1[i * Nc + k] * u2[k * Nc + j];
        temp += u1[i * Nc + k] * u2[k * Nc + j];
      }
      result[i * Nc + j] = temp;
    }
  }
}
__device__ __host__ inline void gaugeAddAssign(Complex *u1, Complex *u2)
{
  u1[0] += u2[0];
  u1[1] += u2[1];
  u1[2] += u2[2];
  u1[3] += u2[3];
  u1[4] += u2[4];
  u1[5] += u2[5];
  u1[6] += u2[6];
  u1[7] += u2[7];
  u1[8] += u2[8];
}
__device__ __host__ inline void gaugeSubAssign(Complex *u1, Complex *u2)
{
  u1[0] -= u2[0];
  u1[1] -= u2[1];
  u1[2] -= u2[2];
  u1[3] -= u2[3];
  u1[4] -= u2[4];
  u1[5] -= u2[5];
  u1[6] -= u2[6];
  u1[7] -= u2[7];
  u1[8] -= u2[8];
}
__device__ __host__ inline void gaugeAssign(Complex *u1, Complex *u2)
{
  u1[0] = u2[0];
  u1[1] = u2[1];
  u1[2] = u2[2];
  u1[3] = u2[3];
  u1[4] = u2[4];
  u1[5] = u2[5];
  u1[6] = u2[6];
  u1[7] = u2[7];
  u1[8] = u2[8];
}
__device__ __host__ inline void gaugeTransposeConj(Complex *u)
{
  // u1[0] = u2[0].conj(); u1[1] = u2[3].conj(); u1[2] = u2[6].conj();
  // u1[3] = u2[1].conj(); u1[4] = u2[4].conj(); u1[5] = u2[7].conj();
  // u1[6] = u2[2].conj(); u1[7] = u2[5].conj(); u1[8] = u2[8].conj();
  Complex temp;
  u[0] = u[0].conj();
  u[4] = u[4].conj();
  u[8] = u[8].conj(); // diag
  temp = u[1];
  u[1] = u[3].conj();
  u[3] = temp.conj();
  temp = u[2];
  u[2] = u[6].conj();
  u[6] = temp.conj();
  temp = u[5];
  u[5] = u[7].conj();
  u[7] = temp.conj();
}

// assume tensor_product is[Ns * Nc * Ns * Nc] and gauge is [Nc * Nc]
__device__ __host__ void tensorProductAddition(Complex *tensor_product, Complex *gauge, int mu, int nu,
                                               Complex co_eff = Complex(1, 0))
{
  if (mu == 0 && nu == 1) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        tensor_product[i * Ns * Nc + j] += co_eff * Complex(0, -1) * gauge[i * Nc + j] * 2;
        tensor_product[Nc * Ns * Nc + i * Ns * Nc + 1 * Nc + j] += co_eff * Complex(0, 1) * gauge[i * Nc + j] * 2;
        tensor_product[2 * Nc * Ns * Nc + i * Ns * Nc + 2 * Nc + j] += co_eff * Complex(0, -1) * gauge[i * Nc + j] * 2;
        tensor_product[3 * Nc * Ns * Nc + i * Ns * Nc + 3 * Nc + j] += co_eff * Complex(0, 1) * gauge[i * Nc + j] * 2;
      }
    }
  } else if (mu == 0 && nu == 2) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        tensor_product[i * Ns * Nc + 1 * Nc + j] += co_eff * Complex(-1, 0) * gauge[i * Nc + j] * 2;
        tensor_product[Nc * Ns * Nc + i * Ns * Nc + j] += co_eff * Complex(1, 0) * gauge[i * Nc + j] * 2;
        tensor_product[2 * Nc * Ns * Nc + i * Ns * Nc + 3 * Nc + j] += co_eff * Complex(-1, 0) * gauge[i * Nc + j] * 2;
        tensor_product[3 * Nc * Ns * Nc + i * Ns * Nc + 2 * Nc + j] += co_eff * Complex(1, 0) * gauge[i * Nc + j] * 2;
      }
    }
  } else if (mu == 0 && nu == 3) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        tensor_product[i * Ns * Nc + 1 * Nc + j] += co_eff * Complex(0, 1) * gauge[i * Nc + j] * 2;
        tensor_product[Nc * Ns * Nc + i * Ns * Nc + j] += co_eff * Complex(0, 1) * gauge[i * Nc + j] * 2;
        tensor_product[2 * Nc * Ns * Nc + i * Ns * Nc + 3 * Nc + j] += co_eff * Complex(0, -1) * gauge[i * Nc + j] * 2;
        tensor_product[3 * Nc * Ns * Nc + i * Ns * Nc + 2 * Nc + j] += co_eff * Complex(0, -1) * gauge[i * Nc + j] * 2;
      }
    }
  } else if (mu == 1 && nu == 2) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        tensor_product[i * Ns * Nc + 1 * Nc + j] += co_eff * Complex(0, -1) * gauge[i * Nc + j] * 2;
        tensor_product[Nc * Ns * Nc + i * Ns * Nc + j] += co_eff * Complex(0, -1) * gauge[i * Nc + j] * 2;
        tensor_product[2 * Nc * Ns * Nc + i * Ns * Nc + 3 * Nc + j] += co_eff * Complex(0, -1) * gauge[i * Nc + j] * 2;
        tensor_product[3 * Nc * Ns * Nc + i * Ns * Nc + 2 * Nc + j] += co_eff * Complex(0, -1) * gauge[i * Nc + j] * 2;
      }
    }
  } else if (mu == 1 && nu == 3) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        tensor_product[i * Ns * Nc + 1 * Nc + j] += co_eff * Complex(-1, 0) * gauge[i * Nc + j] * 2;
        tensor_product[Nc * Ns * Nc + i * Ns * Nc + j] += co_eff * Complex(1, 0) * gauge[i * Nc + j] * 2;
        tensor_product[2 * Nc * Ns * Nc + i * Ns * Nc + 3 * Nc + j] += co_eff * Complex(1, 0) * gauge[i * Nc + j] * 2;
        tensor_product[3 * Nc * Ns * Nc + i * Ns * Nc + 2 * Nc + j] += co_eff * Complex(-1, 0) * gauge[i * Nc + j] * 2;
      }
    }
  } else if (mu == 2 && nu == 3) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        tensor_product[i * Ns * Nc + j] += co_eff * Complex(0, 1) * gauge[i * Nc + j] * 2;
        tensor_product[Nc * Ns * Nc + i * Ns * Nc + 1 * Nc + j] += co_eff * Complex(0, -1) * gauge[i * Nc + j] * 2;
        tensor_product[2 * Nc * Ns * Nc + i * Ns * Nc + 2 * Nc + j] += co_eff * Complex(0, -1) * gauge[i * Nc + j] * 2;
        tensor_product[3 * Nc * Ns * Nc + i * Ns * Nc + 3 * Nc + j] += co_eff * Complex(0, 1) * gauge[i * Nc + j] * 2;
      }
    }
  }
}
#ifdef DEBUG
__device__ __host__ void printGauge(Complex *gauge)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      gauge[i * 3 + j].output();
    }
  }
}
#endif
class Clover {
private:
  // Point point_;
  Complex *first_item;
  Complex *second_item;
  Complex *third_item;
  Complex *fourth_item;
  Complex *dst_; // addr to store
public:
  Clover() = default;
  __device__ __host__ void setDst(Complex *dst) { dst_ = dst; }
  __device__ __host__ Complex *getDst() const { return dst_; }
  __device__ __host__ void getGauge(Complex *gauge, const Point &point, int mu, int nu, int Lx, int Ly, int Lz, int Lt)
  { // U_1
    Point move_point;
    first_item = point.getPointGauge(gauge, mu, Lx, Ly, Lz, Lt); // U_\mu(x)

    move_point = point.move(FRONT, mu, Lx, Ly, Lz, Lt);                // x + \mu
    second_item = move_point.getPointGauge(gauge, nu, Lx, Ly, Lz, Lt); // U_\nu(x+\mu)

    move_point = point.move(FRONT, nu, Lx, Ly, Lz, Lt);               // x + \nu
    third_item = move_point.getPointGauge(gauge, mu, Lx, Ly, Lz, Lt); // U_\mu(x+\nu)

    fourth_item = point.getPointGauge(gauge, nu, Lx, Ly, Lz, Lt); // U_\nu(x)
  }
  __device__ __host__ void F_1(Complex *partial_result, int parity, int i, int j, Point p)
  {
    Complex lhs[Nc * Nc];
    Complex rhs[Nc * Nc];
    gaugeAssign(lhs, first_item);
    gaugeAssign(rhs, second_item);
    gaugeMul(partial_result, lhs, rhs);
    gaugeAssign(lhs, partial_result);
    gaugeAssign(rhs, third_item);
    gaugeTransposeConj(rhs);
    gaugeMul(partial_result, lhs, rhs);
    gaugeAssign(lhs, partial_result);
    gaugeAssign(rhs, fourth_item);
    gaugeTransposeConj(rhs);
    gaugeMul(partial_result, lhs, rhs);
  }
  __device__ __host__ void F_2(Complex *partial_result)
  {
    Complex lhs[Nc * Nc];
    Complex rhs[Nc * Nc];
    gaugeAssign(lhs, second_item);
    gaugeAssign(rhs, third_item);
    gaugeTransposeConj(rhs);
    gaugeMul(partial_result, lhs, rhs);
    gaugeAssign(lhs, partial_result);
    gaugeAssign(rhs, fourth_item);
    gaugeTransposeConj(rhs);
    gaugeMul(partial_result, lhs, rhs);
    gaugeAssign(lhs, partial_result);
    gaugeAssign(rhs, first_item);
    gaugeMul(partial_result, lhs, rhs);
  }
  __device__ __host__ void F_3(Complex *partial_result)
  {
    Complex lhs[Nc * Nc];
    Complex rhs[Nc * Nc];
    gaugeAssign(lhs, third_item);
    gaugeTransposeConj(lhs);
    gaugeAssign(rhs, fourth_item);
    gaugeTransposeConj(rhs);
    gaugeMul(partial_result, lhs, rhs);
    gaugeAssign(lhs, partial_result);
    gaugeAssign(rhs, first_item);
    gaugeMul(partial_result, lhs, rhs);
    gaugeAssign(lhs, partial_result);
    gaugeAssign(rhs, second_item);
    gaugeMul(partial_result, lhs, rhs);
  }
  __device__ __host__ void F_4(Complex *partial_result)
  {
    // __device__ __host__ void F_4(Complex* partial_result, int parity, int i, int j, Point p) {

    Complex lhs[Nc * Nc];
    Complex rhs[Nc * Nc];
    gaugeAssign(lhs, fourth_item);
    gaugeTransposeConj(lhs);
    gaugeAssign(rhs, first_item);
    gaugeMul(partial_result, lhs, rhs);
    gaugeAssign(lhs, partial_result);
    gaugeAssign(rhs, second_item);
    gaugeMul(partial_result, lhs, rhs);
    gaugeAssign(lhs, partial_result);
    gaugeAssign(rhs, third_item);
    gaugeTransposeConj(rhs);
    gaugeMul(partial_result, lhs, rhs);
  }

  __device__ __host__ void cloverCalculate(Complex *gauge, Complex *clover_ptr, const Point &point, int Lx, int Ly,
                                           int Lz, int Lt)
  {
    Complex clover_item[Ns * Nc * Ns * Nc];
    Complex temp[Nc * Nc];
    Complex sum_gauge[Nc * Nc];
    Point move_point;

    setDst(point.getPointClover(clover_ptr, Lx, Ly, Lz, Lt));

    for (int i = 0; i < Ns * Nc * Ns * Nc; i++) {
      clover_item[i].clear2Zero();
    }
    for (int i = 0; i < Ns; i++) {       // mu_
      for (int j = i + 1; j < Ns; j++) { // nu_
        for (int jj = 0; jj < Nc * Nc; jj++) {
          sum_gauge[jj].clear2Zero();
        }
        // F_1
        getGauge(gauge, point, i, j, Lx, Ly, Lz, Lt);
        F_1(temp, point.getParity(), i, j, point);
        gaugeAddAssign(sum_gauge, temp);

        // F_2
        move_point = point.move(BACK, i, Lx, Ly, Lz, Lt);
        getGauge(gauge, move_point, i, j, Lx, Ly, Lz, Lt);
        F_2(temp);
        gaugeAddAssign(sum_gauge, temp);

        // F_3
        move_point = move_point.move(BACK, j, Lx, Ly, Lz, Lt);
        getGauge(gauge, move_point, i, j, Lx, Ly, Lz, Lt);
        F_3(temp);
        gaugeAddAssign(sum_gauge, temp);

        // F_4
        move_point = move_point.move(FRONT, i, Lx, Ly, Lz, Lt);
        getGauge(gauge, move_point, i, j, Lx, Ly, Lz, Lt);
        F_4(temp);
        gaugeAddAssign(sum_gauge, temp);

        gaugeAssign(temp, sum_gauge);
        gaugeTransposeConj(temp);
        gaugeSubAssign(sum_gauge, temp); // A-A.T.conj()

        tensorProductAddition(clover_item, sum_gauge, i, j, Complex(double(-1) / double(16), 0));
      }
    }
    for (int i = 0; i < Ns * Nc * Ns * Nc; i++) {
      dst_[i] = clover_item[i];
    }
  }
};

// generate matrix A and A^-1
__global__ void gpuClover(void *gauge, void *fermion_out, void *clover_ptr, void *invert_ptr, int Lx, int Ly, int Lz,
                          int Lt, int parity)
{

  assert(parity == 0 || parity == 1);
  // __shared__ double shared_output_vec[BLOCK_SIZE * Ns * Nc * 2];
  Lx >>= 1;
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Ly * Lx);
  int z = thread % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;

  Clover clover;
  Point p(x, y, z, t, parity);
  Complex *dst_ptr = p.getPointVector(static_cast<Complex *>(fermion_out), Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc]; // for GPU
  Complex dst_local[Ns * Nc]; // for GPU
  Complex clover_local[Ns * Nc * Ns * Nc];
  Complex invert_local[Ns * Nc * Ns * Nc];

  loadVector(src_local, fermion_out, p, Lx, Ly, Lz, Lt);
  clover.cloverCalculate(static_cast<Complex *>(gauge), static_cast<Complex *>(clover_ptr), p, Lx, Ly, Lz,
                         Lt); // calculate dslash and store them into matrix T
  Complex *clover_addr = clover.getDst();
  for (int i = 0; i < Ns * Nc * Ns * Nc; i++) {
    clover_local[i] = clover_addr[i];
  }
  // generate A = 1 + T     TODO: optimize
  for (int i = 0; i < Ns * Nc; i++) {
    clover_local[i * Ns * Nc + i] += Complex(1, 0);
  }

  // invert A
  inverseMatrix(clover_local, invert_local);

  // A^{-1}dst
  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }
  for (int i = 0; i < Ns * Nc; i++) {
    for (int j = 0; j < Ns * Nc; j++) {
      dst_local[i] += invert_local[i * Ns * Nc + j] * src_local[j];
    }
  }

  // Store dst
  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }
}

__global__ void gpuDslash(void *gauge, void *fermion_in, void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity)
{
  assert(parity == 0 || parity == 1);

  __shared__ double shared_output_vec[BLOCK_SIZE * Ns * Nc * 2];
  Lx >>= 1;

  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Ly * Lx);
  int z = thread % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;

  Complex u_local[Nc * Nc];   // for GPU
  Complex src_local[Ns * Nc]; // for GPU
  Complex dst_local[Ns * Nc]; // for GPU

  Point p(x, y, z, t, parity);
  Point move_point;

  Complex temp;
  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

  // \mu = 1
  loadGauge(u_local, gauge, 0, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 0, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] - src_local[3 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * Complex(0, 1);
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[2 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * Complex(0, 1);
    }
  }

  move_point = p.move(BACK, 0, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, 0, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] + src_local[3 * Nc + j] * Complex(0, 1)) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * Complex(0, -1);
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + src_local[2 * Nc + j] * Complex(0, 1)) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * Complex(0, -1);
    }
  }

  // \mu = 2
  loadGauge(u_local, gauge, 1, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 1, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] + src_local[3 * Nc + j]) * u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[2 * Nc + j]) * u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += -temp;
    }
  }

  move_point = p.move(BACK, 1, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, 1, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] - src_local[3 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += -temp;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + src_local[2 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp;
    }
  }

  // \mu = 3
  loadGauge(u_local, gauge, 2, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 2, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] - src_local[2 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * Complex(0, 1);
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + src_local[3 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * Complex(0, -1);
    }
  }

  move_point = p.move(BACK, 2, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, 2, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] + src_local[2 * Nc + j] * Complex(0, 1)) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * Complex(0, -1);
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[3 * Nc + j] * Complex(0, 1)) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * Complex(0, 1);
    }
  }

  // \mu = 4
  loadGauge(u_local, gauge, 3, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 3, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] - src_local[2 * Nc + j]) * u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += -temp;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[3 * Nc + j]) * u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += -temp;
    }
  }

  move_point = p.move(BACK, 3, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, 3, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] + src_local[2 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + src_local[3 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp;
    }
  }

  // store result
  double *dest = static_cast<double *>(fermion_out) + (blockIdx.x * BLOCK_SIZE) * Ns * Nc * 2;
  double *dest_temp_double = (double *)dst_local;
  for (int i = 0; i < Ns * Nc * 2; i++) {
    shared_output_vec[threadIdx.x * Ns * Nc * 2 + i] = dest_temp_double[i];
  }
  __syncthreads();
  // load to global memory
  for (int i = threadIdx.x; i < BLOCK_SIZE * Ns * Nc * 2; i += BLOCK_SIZE) {
    dest[i] = shared_output_vec[i];
  }
}

void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity)
{
  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];

  int space = Lx * Ly * Lz * Lt >> 1;
  dim3 gridDim(space / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);

  void *clover_matrix;
  void *invert_matrix;
  checkCudaErrors(cudaMalloc(&clover_matrix, sizeof(Complex) * Ns * Nc * Ns * Nc * Lx * Ly * Lz * Lt));
  checkCudaErrors(cudaMalloc(&invert_matrix, sizeof(Complex) * Ns * Nc * Ns * Nc * Lx * Ly * Lz * Lt));

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
  auto start = std::chrono::high_resolution_clock::now();

  void *args[] = {&gauge, &fermion_in, &fermion_out, &Lx, &Ly, &Lz, &Lt, &parity};
  checkCudaErrors(cudaLaunchKernel((void *)gpuDslash, gridDim, blockDim, args));
  checkCudaErrors(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  void *args1[] = {&gauge, &fermion_out, &clover_matrix, &invert_matrix, &Lx, &Ly, &Lz, &Lt, &parity};
  checkCudaErrors(cudaLaunchKernel((void *)gpuClover, gridDim, blockDim, args1));
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(clover_matrix));
  checkCudaErrors(cudaFree(invert_matrix));

  printf("total time: (without malloc free memcpy) : %.9lf sec\n", double(duration) / 1e9);
}