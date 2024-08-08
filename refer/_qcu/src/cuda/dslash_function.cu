#include "include.h"







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
    Complex *U_ptr = d_gauge + pos * Lcc + ((d_parity == 0 )+ 6) * Ltzyxcc -Lzyxcc + (t == 0) * Ltzyxcc ; // 将gau的指针转移到u=3，位置为pos，奇偶为d_parity的位置
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
    U_ptr = d_gauge + pos * Lcc + (d_parity + 6) * Ltzyxcc;//gau需要变换奇偶
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
    U_ptr = d_gauge + pos * Lcc + ((d_parity == 0 )+ 4) * Ltzyxcc -Lyxcc + (z == 0) * Lzyxcc ; // 将gau的指针转移到u=3，位置为pos，奇偶为d_parity的位置
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
    U_ptr = d_gauge + pos * Lcc + (d_parity + 4) * Ltzyxcc;//gau需要变换奇偶
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
    U_ptr = d_gauge + pos * Lcc + ((d_parity == 0 )+ 2) * Ltzyxcc -Lxcc + (y == 0) * Lyxcc ; // 将gau的指针转移到u=3，位置为pos，奇偶为d_parity的位置
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
    U_ptr = d_gauge + pos * Lcc + (d_parity + 2) * Ltzyxcc;//gau需要变换奇偶
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
    U_ptr = d_gauge + pos * Lcc + ((d_parity == 0 )) * Ltzyxcc+ (d_parity == (t+z+y)%2)*( -Lcc + (x == 0) * Lxcc) ; // 将gau的指针转移到u=3，位置为pos，奇偶为d_parity的位置
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
    U_ptr = d_gauge + pos * Lcc + (d_parity ) * Ltzyxcc;//gau需要变换奇偶
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
    for (int i = 0; i < 12; i++){
        Mp_ptr[i] = Mp_buf[i];
    }
}
__global__ void dslash_inner(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity) {

    // int data_length = 6;
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
    //int t = pos / ((Lz-2) * (Ly-2) * (Lx-2))+1,z = pos / ((Ly-2) * (Lx-2)) % (Lz-2) + 1, y = pos / ((Lx-2)) % (Ly-2)+1, x = pos % (Lx-2) + 1;//获取当前位置
    int t = pos / (Lz * Ly * Lx),z = pos / (Ly * Lx) % Lz, y = pos / (Lx) % Ly, x = pos % Lx;//获取当前位置
    pos = t * Lz * Ly * Lx + z * Ly * Lx + y * Lx + x;
    register Complex Mp_buf[12];
    //t-u
    Complex *U_ptr = U_t + U_t_down+ U_parity_change; // 将gau的指针转移到u=3，位置为pos，奇偶为d_parity的位置
    Complex *p_ptr = d_fermi_in + pos * Lsc -Lzyxsc;
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

    for (int i = 0; i < 12; i++)
    {
        Mp_ptr[i] = Mp_buf[i];
    }    


}
__global__ void dslash_clover_inner(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity) {
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

__global__ void dslash_border_revc(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_recive, bool if_border) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int t = pos / (Lz * Ly * Lx),z = pos / (Ly * Lx) % Lz, y = pos / (Lx) % Ly, x = pos % Lx;//获取当前位置
    if((bool(t == 0 || t == Lt-1 || z == 0 || z == Lz-1 || y == 0 || y == Ly-1 || x == 0 || x == Lx-1) ==  if_border) ){
        int data_length = 6;
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


        Complex zero(0, 0), one(1, 0), I(0, 1);
        register Complex Mp_buf[12];
        //t-u
        Complex *U_ptr = U_t + U_t_down+ U_parity_change; // 将gau的指针转移到u=3，位置为pos，奇偶为d_parity的位置
        Complex *p_ptr = d_fermi_in + pos * Lsc -Lzyxsc;
        Complex *Mp_ptr = d_fermi_out + pos * Lsc;
        if(t == 0)
        {
            Complex *data = data_recive + (pos % (Lz * Ly * Lx) + pos / (Lt * Lz * Ly * Lx) * (Lz * Ly * Lx)) * data_length;

            for (int c = 0; c < Lc; c++){
                Mp_buf[c] += data[c];
                Mp_buf[1*Lc + c] += data[1*Lc + c];
                Mp_buf[2*Lc + c] += data[c];
                Mp_buf[3*Lc + c] += data[1*Lc + c];
                //printf("data = %e\n", data[c]);
            }

        }else
        {

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
        }

        //t+u
        if (t == Lt-1)
        {
            U_ptr = U_t + U_parity_nochange;//gau需要变换奇偶
            Complex *data = data_recive + (pos % (Lz * Ly * Lx) + pos / (Lt * Lz * Ly * Lx) + (Lz * Ly * Lx)) * data_length;
            p_ptr = d_fermi_in + pos * Lsc + Lzyxsc - (t == Lt-1) * Ltzyxsc ;
            for (int c = 0; c < Lc; c++){
                Complex buf[2];
                for (int c1 = 0; c1 < Lc; c1++) {
                    buf[0] += U_ptr[c * Lc + c1] * data[c1];
                    buf[1] += U_ptr[c * Lc + c1] * data[c1 + Lc];
                }
                Mp_buf[c] += buf[0];
                Mp_buf[1*Lc + c] += buf[1];
                Mp_buf[2*Lc + c] -= buf[0];
                Mp_buf[3*Lc + c] -= buf[1];
                // printf("data = %e\n", data[0] - p_ptr[0] + p_ptr[2 * Lc + 0]);
            }
        }else{

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
        }

        //z-u
        if(z == 0)
        {
            Complex *data = data_recive + (pos % (Ly * Lx) + pos / (Lz * Ly * Lx) *(Ly * Lx)) * data_length + (Lz*Ly*Lx)*12 ;

            for (int c = 0; c < Lc; c++){
                Mp_buf[c] += data[c];
                Mp_buf[1*Lc + c] += data[1*Lc + c];
                Mp_buf[2*Lc + c] -= data[c]*I;
                Mp_buf[3*Lc + c] += data[1*Lc + c]*I;
                //printf("data = %e\n", data[c]);
            }

        }else
        {
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
        }

        //z+u
        if (z == Lz-1)
        {
            U_ptr = U_z + U_parity_nochange; // gau需要变换奇偶
            Complex *data = data_recive + (pos % (Ly * Lx) + pos / (Lz * Ly * Lx) *(Ly * Lx) + (Lt * Ly * Lx)) * data_length +(Lz*Ly*Lx )*12;
            p_ptr = d_fermi_in + pos * Lsc + Lyxsc - (z == Lz-1) * Lzyxsc ;
            for (int c = 0; c < Lc; c++){
                Complex buf[2];
                for (int c1 = 0; c1 < Lc; c1++) {
                    buf[0] += U_ptr[c * Lc + c1] * data[c1];
                    buf[1] += U_ptr[c * Lc + c1] * data[c1 + Lc];
                }
                Mp_buf[c] += buf[0];
                Mp_buf[1*Lc + c] += buf[1];
                Mp_buf[2*Lc + c] += buf[0]*I;
                Mp_buf[3*Lc + c] -= buf[1]*I;
                // printf("data = %e\n", data[0] - p_ptr[0] + p_ptr[2 * Lc + 0]);
            }
        }else{
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
        }
        //y-u
        if(y == 0)
        {
            Complex *data = data_recive + (pos % (Lx) + pos / (Ly * Lx) *(Lx)) * data_length + (Lz*Ly*Lx + Lt*Ly*Lx)*12 ;

            for (int c = 0; c < Lc; c++){
                Mp_buf[c] += data[c];
                Mp_buf[1*Lc + c] += data[1*Lc + c];
                Mp_buf[2*Lc + c] += data[1*Lc + c];
                Mp_buf[3*Lc + c] -= data[c];
                //printf("data = %e\n", data[c]);
            }

        }else
        {
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
        }
        //y+u
        if (y == Ly-1)
        {
            U_ptr = U_y + U_parity_nochange;//gau需要变换奇偶
            Complex *data = data_recive + (pos % (Lx) + pos / (Ly * Lx) *(Lx) + (Lt*Lz*Lx)) * data_length + (Lz*Ly*Lx + Lt*Ly*Lx)*12 ;
            p_ptr = d_fermi_in + pos * Lsc + Lxsc - (y == Ly-1) * Lyxsc ;
            for (int c = 0; c < Lc; c++){
                Complex buf[2];
                for (int c1 = 0; c1 < Lc; c1++) {
                    buf[0] += U_ptr[c * Lc + c1] * data[c1];
                    buf[1] += U_ptr[c * Lc + c1] * data[c1 + Lc];
                }
                Mp_buf[c] += buf[0];
                Mp_buf[1*Lc + c] += buf[1];
                Mp_buf[2*Lc + c] -= buf[1];
                Mp_buf[3*Lc + c] += buf[0];
            }
        }else{
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
        }
//x-u
        U_ptr = U_x + U_x_down + U_parity_change; // 将gau的指针转移到u=0，位置为pos，奇偶为d_parity的位置
        p_ptr = d_fermi_in + pos * Lsc +(d_parity == (t+z+y)%2)* ( -Lsc + (x == 0) * Lxsc) ;
        if(x == 0 && (d_parity == (t+z+y)%2))
        {
            Complex *data = data_recive + (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx)*12 + ( pos/Lx/2) * data_length;

            for (int c = 0; c < Lc; c++){
                Mp_buf[c] += data[c];
                Mp_buf[1*Lc + c] += data[1*Lc + c];
                Mp_buf[2*Lc + c] -= data[1*Lc + c]*I;
                Mp_buf[3*Lc + c] -= data[c]*I;
                //printf("data = %e\n", data[c]);
                //printf("x=%d, y=%d, z=%d, t=%d, pos = %d ,pos_1 = %d , data = %lf\n", x, y, z, t,  pos, pos/Lx/2, data[0].real);
            }

        }else{
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
                // if(x == 0){
                //     Complex *data = data_recive + (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx)*12 + ( pos / ( Lx)) * data_length;
                //     printf("%lf\t\t%lf\t\t%d\t\t%d\n", data[c].real, buf[0].real, z, ( pos / ( Lx)) * data_length);
                // }
            }
        }
        //x+u

        if (x == Lx-1 && (d_parity != (t+z+y)%2))
        {
            U_ptr = U_x + U_parity_nochange;//gau需要变换奇偶
            Complex *data = data_recive + (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx)*12 + ( pos / ( Lx)/2 + (Lz * Ly * Lt)/2) * data_length;
            p_ptr = d_fermi_in + pos * Lsc + (d_parity != (t+z+y)%2)*(Lsc - (x == Lx-1) * Lxsc );
            for (int c = 0; c < Lc; c++){
                Complex buf[2];
                for (int c1 = 0; c1 < Lc; c1++) {
                    buf[0] += U_ptr[c * Lc + c1] * data[c1];
                    buf[1] += U_ptr[c * Lc + c1] * data[c1 + Lc];
                }
                Mp_buf[c] += buf[0];
                Mp_buf[1*Lc + c] += buf[1];
                Mp_buf[2*Lc + c] += buf[1]*I;
                Mp_buf[3*Lc + c] += buf[0]*I;
                // printf("data = %e\n", data[0] - p_ptr[0] + p_ptr[2 * Lc + 0]);
            }
        }else{
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
        }
        for (int i = 0; i < 12; i++)
        {
            Mp_ptr[i] = Mp_buf[i];
        } 
        // Complex *data = data_recive + (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx)*12 + ( t*Lz*(Ly/2) + z*(Ly/2) + y/2) * data_length;
        // if(x==0)printf("pos = %d , data = %lf\n", pos/Lx, data[0].real);
    }
}


__global__ void output_zero( Complex* d_fermi_out){

    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    d_fermi_out += pos * 12;

    for (int i = 0; i < 12; i++)
    {
        d_fermi_out[i] = 0;
    } 
}



__global__ void dot_gpu_fermi_1(Complex* vector_1, Complex* vector_2, Complex* dot_buf, int Lt, const int Lz, const int Ly, const int Lx){
    /*
    先对s,c方向求和
    */

    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    Complex sum(0, 0);
    vector_1 += pos*Lc*Ls*Lx;
    vector_2 += pos*Lc*Ls*Lx;

    for (int i = 0; i < Lx;i++)
    {
        Complex buf[12];
        for (int j = 0; j < 12;j++){
            buf[j] = vector_2[i * 12 + j];
        }
        for (int j = 0; j < 12;j++){
            sum += vector_1[i * 12 + j].conj() * buf[j];
        }
            
    }
    dot_buf[pos] = sum;
    

}

__global__ void dot_gpu_fermi_2(Complex* dot_buf, int Lt, const int Lz, const int Ly, const int Lx, Complex *dot_result){

    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    Complex zero(0, 0), one(1, 0), I(0, 1);
    Complex sum(0, 0);

    for (int i = 0; i < Lz*Ly ;i++)
    {
        sum += dot_buf[pos*Lz*Ly + i];
    }
    dot_buf[pos*Lz*Ly] = sum;
    __syncthreads();

    sum = zero;
    //x
    if (pos  == 0)
    {
        for (int i = 0; i < Lt; i++)
        {
            sum += dot_buf[i*Lz*Ly];
        }
        dot_result[0] = sum;
        // printf("%e + %ei", dot_result[0].real, dot_result[0].imag);
    }


}

/*
加乘：
a = b + c*kappa
*/
__global__ void add_gpu_fermi(Complex *fermi_a, Complex *fermi_b, Complex *fermi_c, Complex kappa){

    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    fermi_a += pos * 12;
    fermi_b += pos * 12;
    fermi_c += pos * 12;

    // Complex fermi_a_buf[12], fermi_b_buf[12], fermi_c_buf[12]; 

    // for (int i = 0; i < 12; i++)fermi_b_buf[i] = fermi_b[i];
    // for (int i = 0; i < 12; i++)fermi_c_buf[i] = fermi_c[i];

    for (int i = 0; i < 12; i++)
    {
        fermi_a[i] = fermi_b[i] + fermi_c[i] * kappa;
    }   

    // if(pos==0){
    //     printf("add %lf %lf\n", kappa.real, kappa.imag);
    // }

}

/*
dslash用的加乘
a = b - kappa*kapp*a
*/
__global__ void add_gpu_fermi_dslash(Complex *fermi_a, Complex *fermi_b, Complex kappa){

    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    fermi_a += pos * 12;
    fermi_b += pos * 12;


    for (int i = 0; i < 12; i++)
    {
        fermi_a[i] = fermi_b[i] - fermi_a[i] * kappa * kappa;
    }   

}

/*
加乘：
a = b + c*kappa + d*kappa_2
*/
__global__ void add_gpu_fermi_2(Complex *fermi_a, Complex *fermi_b, Complex *fermi_c, Complex *fermi_d, Complex  kappa,  Complex  kappa_2){

    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    fermi_a += pos * 12;
    fermi_b += pos * 12;
    fermi_c += pos * 12;
    fermi_d += pos * 12;

    for (int i = 0; i < 12; i++)
    {
        fermi_a[i] = fermi_b[i] + fermi_c[i] * kappa + fermi_d[i]*kappa_2;
    }   

}

/*
将b赋值到a中
*/
__global__ void fermi_copy(Complex *fermi_a, Complex *fermi_b){

    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    fermi_a += pos * 12;
    fermi_b += pos * 12;

    for (int i = 0; i < 12; i++)
    {
        fermi_a[i] = fermi_b[i];
    }   
}

/*
将b赋值到a中
*/
__global__ void fermi_copy_2(Complex_2 *fermi_a, Complex_2 *fermi_b){

    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    fermi_a += pos * 6;
    fermi_b += pos * 6;

    for (int i = 0; i < 6; i++)
    {
        fermi_a[i] = fermi_b[i];
    }   
}