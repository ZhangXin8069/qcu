#include <sys/types.h>
#pragma nv_verbose
#pragma optimize(5)
#include "qcu.h"
#include <assert.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define even_prt 0
#define odd_prt 1

#define for_ijk_mat for(int i = 0; i < Lc; i++)for(int j = 0; j < Lc; j++)for(int k = 0; k < Lc; k++)

#define Ls 4
#define Lc 3
/*
用于表示U的的位置的指针的宏定义
*/
#define U_t     d_gauge + pos * Lcc + 6 * Ltzyxcc
#define U_z     d_gauge + pos * Lcc + 4 * Ltzyxcc
#define U_y     d_gauge + pos * Lcc + 2 * Ltzyxcc
#define U_x     d_gauge + pos * Lcc
#define U_t_up     Lzyxcc - (t == Lt-1) * Ltzyxcc
#define U_t_down   - Lzyxcc + (t == 0) * Ltzyxcc
#define U_z_up     Lyxcc - (z == Lz-1) * Lzyxcc
#define U_z_down   - Lyxcc + (z == 0) * Lzyxcc
#define U_y_up     Lxcc - (y == Ly-1) * Lyxcc
#define U_y_down   - Lxcc + (y == 0) * Lyxcc
#define U_x_up     (d_parity != (t+z+y)%2)*( Lcc - (x == Lx-1) * Lxcc)
#define U_x_down   (d_parity == (t+z+y)%2)*( -Lcc + (x == 0) * Lxcc) 
#define U_parity_change      (d_parity == 0 ) * Ltzyxcc
#define U_parity_nochange    (d_parity ) * Ltzyxcc

#define checkCudaErrors(err)                                                   \
{                                                                            \
if (err != cudaSuccess) {                                                  \
    fprintf(stderr,                                                          \
            "checkCudaErrors() API error = %04d \"%s\" from file <%s>, "     \
            "line %i.\n",                                                    \
            err, cudaGetErrorString(err), __FILE__, __LINE__);               \
    exit(-1);                                                                \
}                                                                          \
}


class Complex {
public:
    double real;
    double imag;
    __device__ void print(void) {
        printf("%e + %e j\n", this->real, this->imag);
    }
    __device__ Complex(double real = 0.0, double imag = 0.0) {
        this->real = real;
        this->imag = imag;
    }
    __device__ Complex& operator=(const Complex& other) {
        if (this != &other) {
            this->real = other.real;
            this->imag = other.imag;
        }
        return *this;
    }
    __device__ Complex operator=(const double& other) {
        this->real = other;
        this->imag = 0;
        return *this;
    }
    __device__ Complex operator+(const Complex& other) const {
        return Complex(this->real + other.real, this->imag + other.imag);
    }
    __device__ Complex operator-(const Complex& other) const {
        return Complex(this->real - other.real, this->imag - other.imag);
    }
    __device__ Complex operator*(const Complex& other) const {
        return Complex(this->real * other.real - this->imag * other.imag,
            this->real * other.imag + this->imag * other.real);
    }
    __device__ Complex operator*(const double& other) const {
        return Complex(this->real * other, this->imag * other);
    }
    __device__ Complex operator/(const Complex& other) const {
        double denom = other.real * other.real + other.imag * other.imag;
        return Complex((this->real * other.real + this->imag * other.imag) / denom,
            (this->imag * other.real - this->real * other.imag) / denom);
    }
    __device__ Complex operator/(const double& other) const {
        return Complex(this->real / other, this->imag / other);
    }
    __device__ Complex operator-() const { return Complex(-this->real, -this->imag); }
    __device__ Complex& operator+=(const Complex& other) {
        this->real += other.real;
        this->imag += other.imag;
        return *this;
    }
    __device__ Complex& operator-=(const Complex& other) {
        this->real -= other.real;
        this->imag -= other.imag;
        return *this;
    }
    __device__ Complex& operator*=(const Complex& other) {
        this->real = this->real * other.real - this->imag * other.imag;
        this->imag = this->real * other.imag + this->imag * other.real;
        return *this;
    }
    __device__ Complex& operator*=(const double& scalar) {
        this->real *= scalar;
        this->imag *= scalar;
        return *this;
    }
    __device__ Complex& operator/=(const Complex& other) {
        double denom = other.real * other.real + other.imag * other.imag;
        this->real = (real * other.real + imag * other.imag) / denom;
        this->imag = (imag * other.real - real * other.imag) / denom;
        return *this;
    }
    __device__ Complex& operator/=(const double& other) {
        this->real /= other;
        this->imag /= other;
        return *this;
    }
    __device__ bool operator==(const Complex& other) const {
        return (this->real == other.real && this->imag == other.imag);
    }
    __device__ bool operator!=(const Complex& other) const { return !(*this == other); }
    __device__ friend std::ostream& operator<<(std::ostream& os, const Complex& c) {
        if (c.imag >= 0.0) {
            os << c.real << " + " << c.imag << "i";
        }
        else {
            os << c.real << " - " << std::abs(c.imag) << "i";
        }
        return os;
    }
    __device__ Complex conj() { return Complex(this->real, -this->imag); }
};

__device__ void F_uv(Complex *U_u, Complex *U_v, int U_u_up, int U_v_up, int U_u_down, int U_v_down, int par_ch, int par_no, Complex *F)
{
    Complex *f1, *f2, *U_ptr;
    {
        Complex F_buf[18];//用于存放每一个单独的方向
        //1
        U_ptr = U_u + U_v_up + par_ch;//ut(z+)
        f2 = U_v + par_no;//u2()
        f1 = F_buf;
        for_ijk_mat f1[i * Lc + j] += U_ptr[k * Lc + i].conj() * f2[j * Lc + k].conj();
                
        f2 = F_buf + 9;
        U_ptr = U_v + U_u_up + par_ch;//u2(3+)
        for_ijk_mat f2[i * Lc + j] += U_ptr[i * Lc + k] * f1[k * Lc + j];      

        U_ptr = U_u + par_no; // u3()
        for_ijk_mat F[i * Lc + j] += U_ptr[i * Lc + k] * f2[k * Lc + j];
    }
    {
        Complex F_buf[18];//用于存放每一个单独的方向
        //2
        U_ptr = U_v + U_u_down + par_ch; // u2(3-)
        f2 = U_u + U_u_down + par_ch;    // u3(3-)
        f1 = F_buf;
        for_ijk_mat f1[i * Lc + j] += U_ptr[k * Lc + i].conj() * f2[k * Lc + j];

        f2 = F_buf + 9;
        U_ptr = U_u + U_u_down + U_v_up + par_no; // u3(2+3-)
        for_ijk_mat f2[i * Lc + j] += U_ptr[k * Lc + i].conj() * f1[k * Lc + j];

        U_ptr = U_v + par_no; // U2()
        for_ijk_mat F[i * Lc + j] += U_ptr[i * Lc + k] * f2[k * Lc + j];
    }
    {
        Complex F_buf[18];//用于存放每一个单独的方向
        //3
        U_ptr = U_u + U_u_down + U_v_down + par_no; // u3(3-2-)
        f2 = U_v + U_v_down + par_ch;                 // u2(2-)
        f1 = F_buf;
        for_ijk_mat f1[i * Lc + j] += U_ptr[i * Lc + k] * f2[k * Lc + j];

        f2 = F_buf + 9;
        U_ptr = U_v + U_u_down + U_v_down + par_no;  //u2(3-2-)
        for_ijk_mat f2[i * Lc + j] += U_ptr[k * Lc + i].conj() * f1[k * Lc + j];

        U_ptr = U_u + U_u_down + par_ch; // u3(3-)
        for_ijk_mat F[i * Lc + j] += U_ptr[k * Lc + i].conj() * f2[k * Lc + j];
    }
    {
        Complex F_buf[18];//用于存放每一个单独的方向
        //4
        U_ptr = U_v + U_u_up + U_v_down + par_no; // u2(3+2-)
        f2 = U_u + par_no;                 // u3()
        f1 = F_buf;
        for_ijk_mat f1[i * Lc + j] += U_ptr[i * Lc + k] * f2[j * Lc + k].conj();

        f2 = F_buf + 9;
        U_ptr = U_u + U_v_down + par_ch;  //u3(2-)
        for_ijk_mat f2[i * Lc + j] += U_ptr[i * Lc + k] * f1[k * Lc + j];

        U_ptr = U_v + U_v_down + par_ch; // u2(2-)
        for_ijk_mat F[i * Lc + j] += U_ptr[k * Lc + i].conj() * f2[k * Lc + j];
    }
    Complex F_buf[9];
    for (int i = 0; i < Lc; i++)
        for (int j = 0; j < Lc; j++){
            F_buf[i * Lc + j] = F[j * Lc + i].conj();
        }
    for (int i = 0; i < Lc; i++)
        for (int j = 0; j < Lc; j++){
            F[i * Lc + j] -= F_buf[i * Lc + j];
        }
}

__device__ void inv_33(Complex *a, Complex *a_inv)//3*3矩阵求逆
{
    //行列式
    //伴随矩阵求逆
    Complex hang;
    hang = a[0] * (a[4] * a[8] - a[5] * a[7]) + a[1] * (a[5] * a[6] - a[3] * a[8]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
    for (int i = 0; i < Lc; i++)
        for (int j = 0; j < Lc; j++){
            a_inv[i * Lc + j] = a[(i + 1) % Lc * Lc + (j + 1) % Lc] * a[(i + 2) % Lc * Lc + (j + 2) % Lc] - a[(i + 2) % Lc * Lc + (j + 1) % Lc] * a[(i + 1) % Lc * Lc + (j + 2) % Lc];
            a_inv[i * Lc + j] /= hang;
        }

}
__device__ void inv_66(Complex *a, Complex *a_inv)//3*3矩阵求逆,a_inv必须是零矩阵
{
    //高斯矩阵求逆
    for (int i = 0; i < 6; i++){
        a_inv[i * 6 + i] = 1;
    }
    for (int i = 0; i < 6; i++){
        Complex c = a[i * 6 + i];
        for (int k = 0; k < 6; k++){
            a[i * 6 + k] /= c;
            a_inv[i * 6 + k] /= c;
        }
        for (int j = i + 1; j < 6;j++){
            Complex d = a[j * 6 + i];
            for (int k = 0; k < 6; k++){
                a[j * 6 + k] -= a[i * 6 + k] * d;
                a_inv[j * 6 + k] -= a_inv[i * 6 + k] * d;
            }
        }
    }
    // printf("%e",a[0].real);
    for (int i = 5; i >= 0; i--)
    {
        for (int j = i - 1; j >= 0; j--)
        {
            Complex d = a[j * 6 + i];
            for (int k = 0; k < 6; k++){
                a[j * 6 + k] -= a[i * 6 + k] * d;
                a_inv[j * 6 + k] -= a_inv[i * 6 + k] * d;
            }
        }
    }
}
__global__ void dslash(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity) {
    
    int Ltzyxcc = Lt * Lz * Ly * Lx * Lc * Lc;
    int Lzyxcc = Lz * Ly * Lx * Lc * Lc;
    int Lyxcc = Ly * Lx * Lc * Lc;
    int Lxcc = Lx * Lc * Lc;
    int Lcc = Lc * Lc;
    int Ltzyxsc = Lt * Lz * Ly * Lx * Ls * Lc;
    int Lzyxsc = Lz * Ly * Lx * Ls * Lc;
    int Lyxsc = Ly * Lx * Ls * Lc;
    int Lxsc = Lx * Ls * Lc;
    int Lsc = Ls * Lc;
    //int u_pos_max = Ltzyxcc;
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    Complex zero(0, 0), one(1, 0), I(0, 1);
    int t = pos / (Lz * Ly * Lx),z = pos / (Ly * Lx) % Lz, y = pos / (Lx) % Ly, x = pos % Lx;//获取当前位置
    register Complex Mp_buf[12];
    //t-u
    Complex *U_ptr = U_t + U_t_down+ U_parity_change; // 将gau的指针转移到u=3，位置为pos，奇偶为d_parity的位置
    Complex *p_ptr = d_fermi_in + pos * Lsc -Lzyxsc + (t == 0) * Ltzyxsc ;
    Complex *Mp_ptr = d_fermi_out + pos * Lsc;
    for (int c = 0; c < Lc; c++){
        Complex buf[2];//每一行的和暂存
        for (int c1 = 0; c1 < Lc; c1++) {
            buf[0] += U_ptr[c1 * Lc + c].conj() * p_ptr[c1] +
                     U_ptr[c1 * Lc + c].conj() * p_ptr[2 * Lc + c1];

            buf[1] += U_ptr[c1 * Lc + c].conj() * p_ptr[1 * Lc + c1] +
                     U_ptr[c1 * Lc + c].conj() * p_ptr[3 * Lc + c1];           
        }
        Mp_buf[c] += buf[0];
        Mp_buf[1*Lc + c] += buf[1];
        Mp_buf[2*Lc + c] += buf[0];
        Mp_buf[3*Lc + c] += buf[1];
    }

    //t+u
    U_ptr = U_t + U_parity_nochange;//gau需要变换奇偶
    p_ptr = d_fermi_in + pos * Lsc + Lzyxsc - (t == Lt-1) * Ltzyxsc ;
    for (int c = 0; c < Lc; c++){
        Complex buf[2];
        for (int c1 = 0; c1 < Lc; c1++) {
            buf[0] += U_ptr[c * Lc + c1] * p_ptr[c1] -
                      U_ptr[c * Lc + c1] * p_ptr[2 * Lc + c1];

            buf[1] += U_ptr[c * Lc + c1] * p_ptr[1 * Lc + c1] -
                      U_ptr[c * Lc + c1] * p_ptr[3 * Lc + c1];           
        }
        Mp_buf[c] += buf[0];
        Mp_buf[1*Lc + c] += buf[1];
        Mp_buf[2*Lc + c] -= buf[0];
        Mp_buf[3*Lc + c] -= buf[1];
    }
    //z-u
    U_ptr = U_z + U_z_down + U_parity_change; // 将gau的指针转移到u=2，位置为pos，奇偶为d_parity的位置
    p_ptr = d_fermi_in + pos * Lsc -Lyxsc + (z == 0) * Lzyxsc ;
    for (int c = 0; c < Lc; c++){
        Complex buf[2];//每一行的和暂存
        for (int c1 = 0; c1 < Lc; c1++) {
            buf[0] += U_ptr[c1 * Lc + c].conj() * p_ptr[c1] +
                     U_ptr[c1 * Lc + c].conj() * p_ptr[2 * Lc + c1] * I;

            buf[1] += U_ptr[c1 * Lc + c].conj() * p_ptr[1 * Lc + c1] -
                     U_ptr[c1 * Lc + c].conj() * p_ptr[3 * Lc + c1] * I;           
        }
        Mp_buf[c] += buf[0];
        Mp_buf[1*Lc + c] += buf[1];
        Mp_buf[2*Lc + c] -= buf[0]*I;
        Mp_buf[3*Lc + c] += buf[1]*I;
    }

    //z+u
    U_ptr = U_z + U_parity_nochange; // gau需要变换奇偶
    p_ptr = d_fermi_in + pos * Lsc + Lyxsc - (z == Lz-1) * Lzyxsc ;
    for (int c = 0; c < Lc; c++){
        Complex buf[2];
        for (int c1 = 0; c1 < Lc; c1++) {
            buf[0] += U_ptr[c * Lc + c1] * p_ptr[c1] -
                      U_ptr[c * Lc + c1] * p_ptr[2 * Lc + c1] * I;

            buf[1] += U_ptr[c * Lc + c1] * p_ptr[1 * Lc + c1] +
                      U_ptr[c * Lc + c1] * p_ptr[3 * Lc + c1] * I;
        }
        Mp_buf[c] += buf[0];
        Mp_buf[1*Lc + c] += buf[1];
        Mp_buf[2*Lc + c] += buf[0]*I;
        Mp_buf[3*Lc + c] -= buf[1]*I;
    }
        //y-u
    U_ptr = U_y + U_y_down + U_parity_change; // 将gau的指针转移到u=1，位置为pos，奇偶为d_parity的位置
    p_ptr = d_fermi_in + pos * Lsc -Lxsc + (y == 0) * Lyxsc ;
    for (int c = 0; c < Lc; c++){
        Complex buf[2];//每一行的和暂存
        for (int c1 = 0; c1 < Lc; c1++) {
            buf[0] += U_ptr[c1 * Lc + c].conj() * p_ptr[c1] -
                     U_ptr[c1 * Lc + c].conj() * p_ptr[3 * Lc + c1];

            buf[1] += U_ptr[c1 * Lc + c].conj() * p_ptr[1 * Lc + c1] +
                     U_ptr[c1 * Lc + c].conj() * p_ptr[2 * Lc + c1];           
        }
        Mp_buf[c] += buf[0];
        Mp_buf[1*Lc + c] += buf[1];
        Mp_buf[2*Lc + c] += buf[1];
        Mp_buf[3*Lc + c] -= buf[0];
    }

    //y+u
    U_ptr = U_y + U_parity_nochange;//gau需要变换奇偶
    p_ptr = d_fermi_in + pos * Lsc + Lxsc - (y == Ly-1) * Lyxsc ;
    for (int c = 0; c < Lc; c++){
        Complex buf[2];
        for (int c1 = 0; c1 < Lc; c1++) {
            buf[0] += U_ptr[c * Lc + c1] * p_ptr[c1] +
                      U_ptr[c * Lc + c1] * p_ptr[3 * Lc + c1];

            buf[1] += U_ptr[c * Lc + c1] * p_ptr[1 * Lc + c1] -
                      U_ptr[c * Lc + c1] * p_ptr[2 * Lc + c1];
        }
        Mp_buf[c] += buf[0];
        Mp_buf[1*Lc + c] += buf[1];
        Mp_buf[2*Lc + c] -= buf[1];
        Mp_buf[3*Lc + c] += buf[0];
    }
    //x-u
    U_ptr = U_x + U_x_down + U_parity_change; // 将gau的指针转移到u=0，位置为pos，奇偶为d_parity的位置
    p_ptr = d_fermi_in + pos * Lsc +(d_parity == (t+z+y)%2)* ( -Lsc + (x == 0) * Lxsc) ;
    for (int c = 0; c < Lc; c++){
        Complex buf[2];//每一行的和暂存
        for (int c1 = 0; c1 < Lc; c1++) {
            buf[0] += U_ptr[c1 * Lc + c].conj() * p_ptr[c1] +
                     U_ptr[c1 * Lc + c].conj() * p_ptr[3 * Lc + c1] * I;

            buf[1] += U_ptr[c1 * Lc + c].conj() * p_ptr[1 * Lc + c1] +
                     U_ptr[c1 * Lc + c].conj() * p_ptr[2 * Lc + c1] * I;           
        }
        Mp_buf[c] += buf[0];
        Mp_buf[1*Lc + c] += buf[1];
        Mp_buf[2*Lc + c] -= buf[1]*I;
        Mp_buf[3*Lc + c] -= buf[0]*I;
    }

    //x+u
    U_ptr = U_x + U_parity_nochange;//gau需要变换奇偶
    p_ptr = d_fermi_in + pos * Lsc + (d_parity != (t+z+y)%2)*(Lsc - (x == Lx-1) * Lxsc );
    for (int c = 0; c < Lc; c++){
        Complex buf[2];
        for (int c1 = 0; c1 < Lc; c1++) {
            buf[0] += U_ptr[c * Lc + c1] * p_ptr[c1] -
                      U_ptr[c * Lc + c1] * p_ptr[3 * Lc + c1] * I;

            buf[1] += U_ptr[c * Lc + c1] * p_ptr[1 * Lc + c1] -
                      U_ptr[c * Lc + c1] * p_ptr[2 * Lc + c1] * I;
        }
        Mp_buf[c] += buf[0];
        Mp_buf[1*Lc + c] += buf[1];
        Mp_buf[2*Lc + c] += buf[1]*I;
        Mp_buf[3*Lc + c] += buf[0]*I;
    }

    //clover
    //F_tz
    Complex aF1[4][9], aF2[4][9];
    {
        Complex F[9];//存放F_uv
        F_uv(U_t, U_z, U_t_up, U_z_up, U_t_down, U_z_down, U_parity_change, U_parity_nochange, F);
        for (int i = 0; i < 9; i++)
        {
            aF1[0][i] -= F[i] * I;
            aF1[3][i] += F[i] * I;
            aF2[0][i] += F[i] * I;
            aF2[3][i] -= F[i] * I;        
        }
    }
    //F_ty
    {
        Complex F[9];//存放F_uv
        F_uv(U_t, U_y, U_t_up, U_y_up, U_t_down, U_y_down, U_parity_change, U_parity_nochange, F);
        for (int i = 0; i < 9; i++)
        {
            aF1[1][i] += F[i];
            aF1[2][i] -= F[i];
            aF2[1][i] -= F[i];
            aF2[2][i] += F[i];        
        }
    }
    //F_tx
    {
        Complex F[9];//存放F_uv
        F_uv(U_t, U_x, U_t_up, U_x_up, U_t_down, U_x_down, U_parity_change, U_parity_nochange, F);
        for (int i = 0; i < 9; i++)
        {
            aF1[1][i] -= F[i] * I;
            aF1[2][i] -= F[i] * I;
            aF2[1][i] += F[i] * I;
            aF2[2][i] += F[i] * I;        
        }
    }
    //F_zy
    {
        Complex F[9];//存放F_uv
        F_uv(U_z, U_y, U_z_up, U_y_up, U_z_down, U_y_down, U_parity_change, U_parity_nochange, F);
        for (int i = 0; i < 9; i++)
        {
            aF1[1][i] += F[i] * I;
            aF1[2][i] += F[i] * I;
            aF2[1][i] += F[i] * I;
            aF2[2][i] += F[i] * I;        
        }
    }
    //F_zx
    {
        Complex F[9];//存放F_uv
        F_uv(U_z, U_x, U_z_up, U_x_up, U_z_down, U_x_down, U_parity_change, U_parity_nochange, F);
        for (int i = 0; i < 9; i++)
        {
            aF1[1][i] += F[i];
            aF1[2][i] -= F[i];
            aF2[1][i] += F[i];
            aF2[2][i] -= F[i];        
        }
    }
    //F_yx
    {
        Complex F[9];//存放F_uv
        F_uv(U_y, U_x, U_y_up, U_x_up, U_y_down, U_x_down, U_parity_change, U_parity_nochange, F);
        for (int i = 0; i < 9; i++)
        {
            aF1[0][i] += F[i] * I;
            aF1[3][i] -= F[i] * I;
            aF2[0][i] += F[i] * I;
            aF2[3][i] -= F[i] * I;        
        }
    }

    Complex inv_mat1[36],inv_mat2[36];

    {
        Complex A1[36],A2[36];

        for (int k = 0; k < 4; k++)
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    A1[(k > 1) * 18 + i * 6 + (k % 2) * 3 + j] = -aF1[k][i * 3 + j] / 8;
                    A2[(k > 1) * 18 + i * 6 + (k % 2) * 3 + j] = -aF2[k][i * 3 + j] / 8;
                }

        for (int i = 0; i < 6; i++)
        {
            A1[i * 6 + i] += 1;
            A2[i * 6 + i] += 1;
        }
        inv_66(A1, inv_mat1);
        inv_66(A2, inv_mat2);
    }

    for (int i = 0; i < 12; i++)
    {
        Mp_ptr[i] = 0;
    }    

    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6;j++)
        {
            Mp_ptr[i] += inv_mat1[i * 6 + j] * Mp_buf[j];
            Mp_ptr[i + 6] += inv_mat2[i * 6 + j] * Mp_buf[j + 6];
        }
}

void dslashQcu(void* fermion_out, void* fermion_in, void* gauge, QcuParam* param, int parity) {
    int Lx = param->lattice_size[0] >> 1;
    int Ly = param->lattice_size[1];
    int Lz = param->lattice_size[2];
    int Lt = param->lattice_size[3];

    //int d_gauge_len = sizeof(Complex) * 4 * Lt * Lz * Ly * Lx * Lc * Lc;
    //int d_fermi_len = sizeof(Complex) * Lt * Lz * Ly * Lx * Ls * Lc;

    Complex* d_gauge = static_cast<Complex*>(gauge), * d_fermi_in = static_cast<Complex*>(fermion_in), * d_fermi_out = static_cast<Complex*>(fermion_out);
 


    dim3 gridDim(Lx * Ly * Lz * Lt / BLOCK_SIZE);

    dim3 blockDim(BLOCK_SIZE);
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    dslash <<<gridDim, blockDim >>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity);

    cudaError_t err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("total time: (without malloc free memcpy) : %.9lf sec\n",
        double(duration) / 1e9);
    //d_gauge[0].print();


}

void dslashCloverQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity){};
void mpiDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, QcuParam *grid){};
void mpiBistabCgQcu(void *gauge, QcuParam *param, QcuParam *grid){};
void ncclDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, QcuParam *grid){};
void ncclBistabCgQcu(void *gauge, QcuParam *param, QcuParam *grid){};