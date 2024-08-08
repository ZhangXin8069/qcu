#include "include.h"



__global__ void dslash_tborder(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_send) {
    
    int Ltzyxcc = Lt * Lz * Ly * Lx * Lc * Lc;
    int Lzyxcc = Lz * Ly * Lx * Lc * Lc;
    // int Lyxcc = Ly * Lx * Lc * Lc;
    // int Lxcc = Lx * Lc * Lc;
    int Lcc = Lc * Lc;
    int Ltzyxsc = Lt * Lz * Ly * Lx * Ls * Lc;
    int Lzyxsc = Lz * Ly * Lx * Ls * Lc;
    // int Lyxsc = Ly * Lx * Ls * Lc;
    // int Lxsc = Lx * Ls * Lc;
    int Lsc = Ls * Lc;
    //int u_pos_max = Ltzyxcc;
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    Complex zero(0, 0), one(1, 0), I(0, 1);
    int t = 0;//, z = pos / (Ly * Lx) % Lz, y = pos / (Lx) % Ly, x = pos % Lx; // 获取当前位置
    register Complex Mp_buf[12];
    //t-u
    Complex *U_ptr = U_t + U_t_down+ U_parity_change; // 将gau的指针转移到u=3，位置为pos，奇偶为d_parity的位置
    Complex *p_ptr = d_fermi_in + pos * Lsc -Lzyxsc + (t == 0) * Ltzyxsc ;
    Complex Mp_ptr[6];
    Complex *data = data_send + pos * 6;
    Complex *data1 = data_send + pos * 6 + (Lz * Ly * Lx) * 6;
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

    }
        //t+u
    pos = pos + (Lt - 1) * Lz * Ly * Lx;
    t = Lt-1;
    p_ptr = d_fermi_in + pos * Lsc + Lzyxsc - (t == Lt-1) * Ltzyxsc ;
    for (int c1 = 0; c1 < Lc; c1++) {
        Mp_buf[6 + c1]         = p_ptr[c1]          - p_ptr[2 * Lc + c1];

        Mp_buf[6 + 1*Lc + c1]  = p_ptr[1 * Lc + c1] - p_ptr[3 * Lc + c1];           
    }


    for (int i = 0; i < 6; i++)
    {
        data[i] = Mp_buf[i];
        data1[i] = Mp_buf[i + 6];
    }    


}

__global__ void dslash_xborder(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_send) {

    int Ltzyxcc = Lt * Lz * Ly * Lx * Lc * Lc;
    int Lxcc = Lx * Lc * Lc;
    int Lcc = Lc * Lc;
    // int Ltzyxsc = Lt * Lz * Ly * Lx * Ls * Lc;
    int Lxsc = Lx * Ls * Lc;
    int Lsc = Ls * Lc;
    //int u_pos_max = Ltzyxcc;
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    Complex zero(0, 0), one(1, 0), I(0, 1);
    int x = 0, t = pos / (Lz * Ly) % Lt, z = pos / (Ly) % Lz, y = pos % Ly; // 获取当前位置
    int pos_t = (t*Lz*Ly + z*Ly + y)/2;
    // pos_t = 3;
    pos = t*Lz*Ly*Lx + z*Ly*Lx + y*Lx + x;
    register Complex Mp_buf[12];

    //x-u
    Complex *U_ptr = U_x + U_x_down + U_parity_change; // 将gau的指针转移到u=0，位置为pos，奇偶为d_parity的位置
    Complex *p_ptr = d_fermi_in + pos * Lsc +(d_parity == (t+z+y)%2)* ( -Lsc + (x == 0) * Lxsc) ;
    Complex Mp_ptr[6];
    Complex *data = data_send + pos_t * 6;
    // Complex *data1 = data_send + pos_t * 6 + (Lz * Ly * Lt)/2 * 6;
    
    if(d_parity == (t+z+y)%2 ){
        
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
            //printf("%lf\t\t%d\t\t%d\n", Mp_buf[1*Lc + c].real, z, y);
        }
        // for (int i = 0; i < 6; i++)
        // {
        //     data[i] = Mp_buf[i];
        // }    
    }
    else{
        //x+u
        pos = pos + (Lx - 1);
        x = Lx-1;
        p_ptr = d_fermi_in + pos * Lsc + (d_parity != (t+z+y)%2)*(Lsc - (x == Lx-1) * Lxsc );
        for (int c1 = 0; c1 < Lc; c1++) {
            Mp_buf[6 + c1]         = p_ptr[c1]          - p_ptr[3 * Lc + c1]*I;

            Mp_buf[6 + 1*Lc + c1]  = p_ptr[1 * Lc + c1] - p_ptr[2 * Lc + c1]*I;           
        }

        
        // for (int i = 0; i < 6; i++)
        // {
        //     data1[i] = Mp_buf[i + 6];
        // }    
    }
    data = data_send + pos_t * 6 + (Lz * Ly * Lt)/2 * 6 * (d_parity != (t+z+y)%2);
    

    for (int i = 0; i < 6; i++)
    {
        data[i] = Mp_buf[i + 6*(d_parity != (t+z+y)%2)];
    }    
    // printf("x=%d, y=%d, z=%d, t=%d, pos=%d, pos_1=%d, data=%lf\n", x, y, z, t, t*Lz*Ly + z*Ly + y,pos_t, data[0].real);

}

__global__ void dslash_yborder(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_send) {
    
    int Ltzyxcc = Lt * Lz * Ly * Lx * Lc * Lc;
    int Lyxcc = Ly * Lx * Lc * Lc;
    int Lxcc = Lx * Lc * Lc;
    int Lcc = Lc * Lc;
    int Lyxsc = Ly * Lx * Ls * Lc;
    int Lxsc = Lx * Ls * Lc;
    int Lsc = Ls * Lc;
    //int u_pos_max = Ltzyxcc;
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    Complex zero(0, 0), one(1, 0), I(0, 1);
    int y = 0, t = pos / (Lx * Lz) % Lt, z = pos / (Lx) % Lz, x = pos % Lx; // 获取当前位置
    int pos_t = pos;
    pos = t*Lz*Ly*Lx + z*Ly*Lx + y*Lx + x;
    register Complex Mp_buf[12];
    //y-u
    Complex *U_ptr = U_y + U_y_down + U_parity_change; // 将gau的指针转移到u=1，位置为pos，奇偶为d_parity的位置
    Complex *p_ptr = d_fermi_in + pos * Lsc -Lxsc + (y == 0) * Lyxsc ;
    Complex Mp_ptr[6];
    Complex *data = data_send + pos_t * 6;
    Complex *data1 = data_send + pos_t * 6 + (Lx * Lz * Lt) * 6;
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
        //printf("%lf\t\t%d\t\t%d\n", Mp_buf[1*Lc + c].real, z, y);
    }
    
    //y+u
    pos = pos + (Ly - 1) * Lx;
    y = Ly-1;
    p_ptr = d_fermi_in + pos * Lsc + Lxsc - (y == Ly-1) * Lyxsc ;
    for (int c1 = 0; c1 < Lc; c1++) {
        Mp_buf[6 + c1]         = p_ptr[c1]          + p_ptr[3 * Lc + c1];

        Mp_buf[6 + 1*Lc + c1]  = p_ptr[1 * Lc + c1] - p_ptr[2 * Lc + c1];           
    }
    //if(x==0 && y==Ly-1 && t==3 && z==1)printf("data=%e\n",Mp_buf[6].real);

    for (int i = 0; i < 6; i++)
    {
        data[i] = Mp_buf[i];
        data1[i] = Mp_buf[i + 6];
    }    


}
__global__ void dslash_zborder(Complex* d_gauge, Complex* d_fermi_in, Complex* d_fermi_out, int Lt, const int Lz, const int Ly, const int Lx, const int d_parity, Complex* data_send) {
    
    int Ltzyxcc = Lt * Lz * Ly * Lx * Lc * Lc;
    int Lzyxcc = Lz * Ly * Lx * Lc * Lc;
    int Lyxcc = Ly * Lx * Lc * Lc;
    int Lcc = Lc * Lc;
    int Lzyxsc = Lz * Ly * Lx * Ls * Lc;
    int Lyxsc = Ly * Lx * Ls * Lc;
    int Lsc = Ls * Lc;
    //int u_pos_max = Ltzyxcc;
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    Complex zero(0, 0), one(1, 0), I(0, 1);
    int z = 0, t = pos / (Lx * Ly) % Lt, y = pos / (Lx) % Ly, x = pos % Lx; // 获取当前位置
    int pos_t = pos;
    pos = t*Lz*Ly*Lx + z*Ly*Lx + y*Lx + x;
    register Complex Mp_buf[12];
    //z-u
    Complex *U_ptr = U_z + U_z_down + U_parity_change; // 将gau的指针转移到u=2，位置为pos，奇偶为d_parity的位置
    Complex *p_ptr = d_fermi_in + pos * Lsc -Lyxsc + (z == 0) * Lzyxsc ;
    Complex Mp_ptr[6];
    Complex *data = data_send + pos_t * 6;
    Complex *data1 = data_send + pos_t * 6 + (Lx * Ly * Lt) * 6;
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
        //printf("%lf\t\t%d\t\t%d\n", Mp_buf[1*Lc + c].real, z, y);
    }
    
    //z+u
    pos = pos + (Lz - 1) * Ly * Lx;
    z = Lz-1;
    p_ptr = d_fermi_in + pos * Lsc + Lyxsc - (z == Lz-1) * Lzyxsc ;
    for (int c1 = 0; c1 < Lc; c1++) {
        Mp_buf[6 + c1]         = p_ptr[c1]          - p_ptr[2 * Lc + c1]*I;

        Mp_buf[6 + 1*Lc + c1]  = p_ptr[1 * Lc + c1] + p_ptr[3 * Lc + c1]*I;           
    }


    for (int i = 0; i < 6; i++)
    {
        data[i] = Mp_buf[i];
        data1[i] = Mp_buf[i + 6];
    }    


}