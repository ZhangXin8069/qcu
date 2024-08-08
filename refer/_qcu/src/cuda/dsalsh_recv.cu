#include "include.h"



__global__ void dslash_tborder_rec(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_recive) {
    
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
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    Complex zero(0, 0), one(1, 0), I(0, 1);
    int t = 0, z = pos / (Ly * Lx) % Lz, y = pos / (Lx) % Ly, x = pos % Lx; // 获取当前位置
    register Complex Mp_buf[12];

    //t=0
    
    Complex *U_ptr = U_t + U_t_down+ U_parity_change; // 将gau的指针转移到u=3，位置为pos，奇偶为d_parity的位置
    Complex *p_ptr = d_fermi_in + pos * Lsc -Lzyxsc;
    Complex *Mp_ptr = d_fermi_out + pos * Lsc;

    //t-u
    Complex *data = data_recive + (pos % (Lz * Ly * Lx) + pos / (Lt * Lz * Ly * Lx) * (Lz * Ly * Lx)) * data_length;
    for (int c = 0; c < Lc; c++){
        Mp_buf[c] += data[c];
        Mp_buf[1*Lc + c] += data[1*Lc + c];
        Mp_buf[2*Lc + c] += data[c];
        Mp_buf[3*Lc + c] += data[1*Lc + c];
        //printf("data = %e\n", data[c]);
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
    for (int i = 0; i < 12; i++){

        Mp_ptr[i] = Mp_buf[i];
    }   



    //另一侧t=Lt-1
    for (int i = 0; i < 12; i++){
        Mp_buf[i] = 0;
    }    

    pos = pos + (Lt - 1) * Lz * Ly * Lx;//切换位置到另一边
    t = Lt-1;
    //t-u
    U_ptr = U_t + U_t_down+ U_parity_change; // 将gau的指针转移到u=3，位置为pos，奇偶为d_parity的位置
    p_ptr = d_fermi_in + pos * Lsc -Lzyxsc;
    Mp_ptr = d_fermi_out + pos * Lsc;

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
    Mp_ptr = d_fermi_out + pos * Lsc;
    data = data_recive + (pos % (Lz * Ly * Lx) + pos / (Lt * Lz * Ly * Lx) + (Lz * Ly * Lx)) * data_length;
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
        Mp_buf[3 * Lc + c] -= buf[1];
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

    for (int i = 0; i < 12; i++){
        Mp_ptr[i] = Mp_buf[i];
    }    

}

__global__ void dslash_xborder_rec(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_recive) {
    
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
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    Complex zero(0, 0), one(1, 0), I(0, 1);
    int x = 0, t = pos / (Lz * Ly) % Lt, z = pos / (Ly) % Lz, y = pos % Ly; // 获取当前位置
    pos = t*Lz*Ly*Lx + z*Ly*Lx + y*Lx + x;
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
    //x=Lx-1
    for (int i = 0; i < 12; i++){
        Mp_buf[i] = 0;
    }    
    pos = pos + (Lx - 1);
    x = Lx-1;

        //t-u
    U_ptr = U_t + U_t_down+ U_parity_change; // 将gau的指针转移到u=3，位置为pos，奇偶为d_parity的位置
    p_ptr = d_fermi_in + pos * Lsc -Lzyxsc;
    Mp_ptr = d_fermi_out + pos * Lsc;
    
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


}

__global__ void dslash_yborder_rec(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_recive) {
    
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
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    Complex zero(0, 0), one(1, 0), I(0, 1);
    int y = 0, t = pos / (Lx * Lz) % Lt, z = pos / (Lx) % Lz, x = pos % Lx; // 获取当前位置
    //int pos_t = pos;
    pos = t*Lz*Ly*Lx + z*Ly*Lx + y*Lx + x;
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
    Complex *data = data_recive + (pos % (Lx) + pos / (Ly * Lx) *(Lx)) * data_length + (Lz*Ly*Lx + Lt*Ly*Lx)*12 ;

    for (int c = 0; c < Lc; c++){
        Mp_buf[c] += data[c];
        Mp_buf[1*Lc + c] += data[1*Lc + c];
        Mp_buf[2*Lc + c] += data[1*Lc + c];
        Mp_buf[3*Lc + c] -= data[c];
        //printf("data = %e\n", data[c]);
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


    for (int i = 0; i < 12; i++)
    {
        Mp_buf[i] = 0;
    }    

        //y+u
    pos = pos + (Ly - 1) * Lx;
    y = Ly-1;

        //t-u
    U_ptr = U_t + U_t_down+ U_parity_change; // 将gau的指针转移到u=3，位置为pos，奇偶为d_parity的位置
    p_ptr = d_fermi_in + pos * Lsc -Lzyxsc;
    Mp_ptr = d_fermi_out + pos * Lsc;
    
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
    data = data_recive + (pos % (Lx) + pos / (Ly * Lx) *(Lx) + (Lt*Lz*Lx)) * data_length + (Lz*Ly*Lx + Lt*Ly*Lx)*12 ;
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
__global__ void dslash_zborder_rec(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_recive) {

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
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    Complex zero(0, 0), one(1, 0), I(0, 1);
    int z = 0, t = pos / (Lx * Ly) % Lt, y = pos / (Lx) % Ly, x = pos % Lx; // 获取当前位置
    // int pos_t = pos;
    pos = t*Lz*Ly*Lx + z*Ly*Lx + y*Lx + x;
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
    {
        Complex *data = data_recive + (pos % (Ly * Lx) + pos / (Lz * Ly * Lx) *(Ly * Lx)) * data_length + (Lz*Ly*Lx)*12 ;

        for (int c = 0; c < Lc; c++){
            Mp_buf[c] += data[c];
            Mp_buf[1*Lc + c] += data[1*Lc + c];
            Mp_buf[2*Lc + c] -= data[c]*I;
            Mp_buf[3*Lc + c] += data[1*Lc + c]*I;
            //printf("data = %e\n", data[c]);
        }
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


    //清零
    for (int i = 0; i < 12; i++)
    {
        Mp_buf[i] = 0;
    }    
    //
    pos = pos + (Lz - 1) * Ly * Lx;
    z = Lz-1;

    //t-u
    U_ptr = U_t + U_t_down+ U_parity_change; // 将gau的指针转移到u=3，位置为pos，奇偶为d_parity的位置
    p_ptr = d_fermi_in + pos * Lsc -Lzyxsc;
    Mp_ptr = d_fermi_out + pos * Lsc;
    
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




__global__ void dslash_side_revc(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_recive, bool if_border) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int t = pos / (Lz * Ly * Lx),z = pos / (Ly * Lx) % Lz, y = pos / (Lx) % Ly, x = pos % Lx;//获取当前位置
    if(((t == 0) + (t == Lt-1) + (z == 0) + (z == Lz-1) + (y == 0) + (y == Ly-1) + (x == 0) + (x == Lx-1)) >  1){
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
