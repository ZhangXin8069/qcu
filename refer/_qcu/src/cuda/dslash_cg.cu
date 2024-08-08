/********************************
 * 
 * 进行cuda的cg
 * 尝试shared_memery
 * 完成：
 * fermi矩阵多线程求和
 * 将函数封装在class内
 * 
 * 
 * 
 * 
*/
#include "include.h"
#include <cstdio>
#include <cstdlib>

//#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define even_prt 0
#define odd_prt 1

#define for_ijk_mat for(int i = 0; i < Lc; i++)for(int j = 0; j < Lc; j++)for(int k = 0; k < Lc; k++)
using data_type = cuDoubleComplex;

class Dslash_class{
public:
    /**************存放变量*****************/
    int Lx, Ly, Lz, Lt;
    Complex *d_gauge, *d_fermi_in, *d_fermi_out;
    Complex* dot_buf;//求和临时变量
    Complex *data_send, *data_send_host_t, *data_send_host_x, *data_send_host_z, *data_send_host_y;//需要发送的数据
    Complex *data_recive, *data_recive_host_tdown, *data_recive_host_tup, *data_recive_host_xdown, *data_recive_host_xup,  *data_recive_host_zdown, *data_recive_host_zup, *data_recive_host_ydown, *data_recive_host_yup;//mpi接收数据地址
    cudaStream_t stream_copy, stream_compute, stream_copy_t, stream_copy_x, stream_copy_y, stream_copy_z;//cuda流
    Complex *dot_result, *dot_mpi_buf;//存储点乘后的结果使用统一内存地址
    int data_length_t, data_length_x, data_length_z, data_length_y;
    int my_rank = 0, my_size = 0;//mpi当前进程地址
    int BLOCK_SIZE = 8;
    int gpu_assignment[4] = {1,1,1,2};//多卡并行的切法
    int x, y, z, t;//mpi中当前块的的位置
    int if_init = 0;
    bool parity;
    MPI_Request request, request_tup, request_tdown, request_xup, request_xdown, request_zup, request_zdown, request_yup, request_ydown;
    MPI_Status status;
    MPI_Request *request_dot;//存放点乘结果mpi传递状态
    ncclComm_t comm;
    ncclUniqueId id;
            

public:

    /*
    初始化函数
    功能：
    1.为class中参数赋值   
    2.分配mpi数据分发所需要的显存和内存
    */
    void init(int Lt_input,int Lz_input, int Ly_input, int Lx_input){
        this->Lt = Lt_input;
        this->Lz = Lz_input;
        this->Ly = Ly_input;
        this->Lx = Lx_input;

        this->data_length_t = Lz * Ly * Lx * 6;
        this->data_length_x = Lz * Ly * Lt * 6 / 2;
        this->data_length_z = Lx * Ly * Lt * 6;
        this->data_length_y = Lx * Lz * Lt * 6;

        //确定当前mpi的位置
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &my_size);

        //nccl
        if (my_rank == 0) ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        ncclCommInitRank(&comm, my_size, id, my_rank);

        //为点乘结果mpi传递状态分配内存
        cudaMallocHost(&request_dot, my_size * sizeof(MPI_Request));

        //创建CUDA流
        cudaStreamCreate(&stream_copy);
        cudaStreamCreate(&stream_compute);

        //分配空间用来存放求内积时的临时变量
        cudaMalloc(&dot_buf, Lt * Ly * Lz * Lx * sizeof(Complex));
        cudaMallocManaged(&dot_result, 3 * sizeof(Complex));
        
        //需要发送的数据分配显存和内存
        cudaMalloc(&data_send, (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx + Lt*Lz*Ly)*6 * sizeof(Complex) * 2);//
        cudaMallocHost(&data_send_host_t, data_length_t * sizeof(Complex) * 2);//
        cudaMallocHost(&data_send_host_x, data_length_x * sizeof(Complex) * 2);//
        cudaMallocHost(&data_send_host_y, data_length_y * sizeof(Complex) * 2);//
        cudaMallocHost(&data_send_host_z, data_length_z * sizeof(Complex) * 2);//

        //接收的数据分配显存和内存
        cudaMalloc(&data_recive, (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx + Lt*Lz*Ly)*6 * sizeof(Complex) * 2);//
        cudaMallocHost(&data_recive_host_tup, data_length_t * sizeof(Complex));//
        cudaMallocHost(&data_recive_host_tdown, data_length_t * sizeof(Complex));//
        cudaMallocHost(&data_recive_host_xup, data_length_x * sizeof(Complex));//
        cudaMallocHost(&data_recive_host_xdown, data_length_x * sizeof(Complex));//
        cudaMallocHost(&data_recive_host_yup, data_length_y * sizeof(Complex));//
        cudaMallocHost(&data_recive_host_ydown, data_length_y * sizeof(Complex));//
        cudaMallocHost(&data_recive_host_zup, data_length_z * sizeof(Complex));//
        cudaMallocHost(&data_recive_host_zdown, data_length_z * sizeof(Complex));//

        //找到当前mpi块的位置
        x = my_rank % (gpu_assignment[3] * gpu_assignment[2] * gpu_assignment[1] * gpu_assignment[0]) / (gpu_assignment[3] * gpu_assignment[2] * gpu_assignment[1]);
        y = my_rank % (gpu_assignment[3] * gpu_assignment[2] * gpu_assignment[1]) / (gpu_assignment[3] * gpu_assignment[2]);
        z = my_rank % (gpu_assignment[3] * gpu_assignment[2]) / (gpu_assignment[3]);
        t = my_rank % gpu_assignment[3];

        
        cudaMallocManaged(&dot_mpi_buf, my_size * sizeof(Complex));

        //标记初始化完成
        if_init = 1;
    }

    /*
    dslash乘法
    功能：
    U矩阵与fermi_in向量的乘法，结果放在fermi_out中
    */
    void dslash_multiply(Complex *gauge, Complex *fermi_out, Complex *fermi_in, bool parity_input){
        // if (if_init == 0)
        // {
        //     printf("ERROR: dslash_gpu还未初始化!");
        //     exit(0);
        // }

        //参数输入
        parity = parity_input;
        d_gauge = gauge;
        d_fermi_in = fermi_in;
        d_fermi_out = fermi_out;

        //计算需要传输的边界
        dslash_tborder <<<Lx * Ly * Lz  / BLOCK_SIZE, BLOCK_SIZE >>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_send);//计算t方向的边界
        dslash_xborder <<<Lt * Ly * Lz  / BLOCK_SIZE, BLOCK_SIZE >>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_send + (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx)*12);//计算x方向的边界 
        dslash_zborder <<<Lt * Ly * Lx  / BLOCK_SIZE, BLOCK_SIZE >>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_send + (Lz*Ly*Lx)*12);//计算z方向的边界
        dslash_yborder <<<Lt * Lz * Lx  / BLOCK_SIZE, BLOCK_SIZE >>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_send + (Lz*Ly*Lx + Lt*Ly*Lx)*12);//计算y方向的边界
        
        //非阻塞传递将内容传至cpu
        cudaMemcpyAsync(data_send_host_t, data_send, data_length_t * sizeof(Complex) * 2, cudaMemcpyDeviceToHost, stream_copy);
        cudaMemcpyAsync(data_send_host_x, data_send + (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx)*12, data_length_x * sizeof(Complex) * 2, cudaMemcpyDeviceToHost, stream_copy);//copy x
        cudaMemcpyAsync(data_send_host_z, data_send + (Lz*Ly*Lx)*12, data_length_z * sizeof(Complex) * 2, cudaMemcpyDeviceToHost, stream_copy);//copy z
        cudaMemcpyAsync(data_send_host_y, data_send + (Lz*Ly*Lx + Lt*Ly*Lx)*12, data_length_y * sizeof(Complex) * 2, cudaMemcpyDeviceToHost, stream_copy);//copy y

        //计算非边缘的乘法内容
        dslash_inner <<<Lx * Ly  * (Lz ) * (Lt)  / BLOCK_SIZE, BLOCK_SIZE , 0, stream_compute>>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity);

        //等待内部dslash完成
        cudaStreamSynchronize(stream_copy);

        //mpi开始传输(cpu->cpu)
        int add;
        //t_up
        MPI_Isend(data_send_host_t,data_length_t * 2,MPI_DOUBLE,my_rank + 1 - (t == (gpu_assignment[3]-1))*gpu_assignment[3],99,MPI_COMM_WORLD, &request);
        MPI_Irecv(data_recive_host_tup,data_length_t * 2,MPI_DOUBLE,my_rank - 1 + (t == 0)*gpu_assignment[3],99,MPI_COMM_WORLD, &request_tup);
        //t_down
        MPI_Isend(data_send_host_t + data_length_t, data_length_t * 2,MPI_DOUBLE,my_rank - 1 + (t == 0)*gpu_assignment[3],98,MPI_COMM_WORLD, &request);
        MPI_Irecv(data_recive_host_tdown,data_length_t * 2,MPI_DOUBLE,my_rank + 1 - (t == (gpu_assignment[3]-1))*gpu_assignment[3],98,MPI_COMM_WORLD, &request_tdown);
        //x_up
        add = gpu_assignment[3] * gpu_assignment[2] * gpu_assignment[1];
        my_size = gpu_assignment[3] * gpu_assignment[2] * gpu_assignment[1] * gpu_assignment[0];
        MPI_Isend(data_send_host_x,data_length_x * 2,MPI_DOUBLE,my_rank + add - (x == (gpu_assignment[0]-1))*my_size,97,MPI_COMM_WORLD, &request);
        MPI_Irecv(data_recive_host_xup,data_length_x * 2,MPI_DOUBLE,my_rank - add + (x == 0)*my_size,97,MPI_COMM_WORLD, &request_xup);
        //x_down
        MPI_Isend(data_send_host_x + data_length_x,data_length_x * 2,MPI_DOUBLE,my_rank - add + (x == 0)*my_size,96,MPI_COMM_WORLD, &request);
        MPI_Irecv(data_recive_host_xdown,data_length_x * 2,MPI_DOUBLE,my_rank + add - (x == (gpu_assignment[0]-1))*my_size,96,MPI_COMM_WORLD, &request_xdown);
        //y_up
        add = gpu_assignment[3] * gpu_assignment[2];
        my_size = gpu_assignment[3] * gpu_assignment[2] * gpu_assignment[1];
        MPI_Isend(data_send_host_y,data_length_y * 2,MPI_DOUBLE,my_rank + add - (y == (gpu_assignment[1]-1))*my_size,93,MPI_COMM_WORLD, &request);
        MPI_Irecv(data_recive_host_yup,data_length_y * 2,MPI_DOUBLE,my_rank - add + (y == 0)*my_size,93,MPI_COMM_WORLD, &request_yup);
        //y_down
        MPI_Isend(data_send_host_y + data_length_y,data_length_y * 2,MPI_DOUBLE,my_rank - add + (y == 0)*my_size,92,MPI_COMM_WORLD, &request);
        MPI_Irecv(data_recive_host_ydown,data_length_y * 2,MPI_DOUBLE,my_rank + add - (y == (gpu_assignment[1]-1))*my_size,92,MPI_COMM_WORLD, &request_ydown);
        //z_up
        add = gpu_assignment[3];
        my_size = gpu_assignment[3] * gpu_assignment[2];
        MPI_Isend(data_send_host_z,data_length_z * 2,MPI_DOUBLE,my_rank + add - (z == (gpu_assignment[2]-1))*my_size,95,MPI_COMM_WORLD, &request);
        MPI_Irecv(data_recive_host_zup,data_length_z * 2,MPI_DOUBLE,my_rank - add + (z == 0)*my_size,95,MPI_COMM_WORLD, &request_zup);
        //z_down
        MPI_Isend(data_send_host_z + data_length_z,data_length_z * 2,MPI_DOUBLE,my_rank - add + (z == 0)*my_size,94,MPI_COMM_WORLD, &request);
        MPI_Irecv(data_recive_host_zdown,data_length_z * 2,MPI_DOUBLE,my_rank + add - (z == (gpu_assignment[2]-1))*my_size,94,MPI_COMM_WORLD, &request_zdown);


        //等待传输完成
        MPI_Wait(&request_tdown, &status);
        MPI_Wait(&request_tup, &status);
        MPI_Wait(&request_xdown, &status);
        MPI_Wait(&request_xup, &status);
        MPI_Wait(&request_zdown, &status);
        MPI_Wait(&request_zup, &status);
        MPI_Wait(&request_ydown, &status);
        MPI_Wait(&request_yup, &status);

        cudaStreamSynchronize(stream_compute);//等待内部dslash完成

        //接收并计算边缘的乘法
        //x
        cudaMemcpyAsync(data_recive + (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx)*12, data_recive_host_xup, data_length_x * sizeof(Complex), cudaMemcpyHostToDevice, stream_copy);
        cudaMemcpyAsync(data_recive + (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx)*12 + data_length_x, data_recive_host_xdown, data_length_x * sizeof(Complex), cudaMemcpyHostToDevice, stream_copy);
        
        cudaStreamSynchronize(stream_copy);//等待内部dslash完成
        dslash_xborder_rec <<<Lz * Ly * Lt/ BLOCK_SIZE, BLOCK_SIZE , 0, stream_compute>>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_recive);
        //t
        cudaMemcpyAsync(data_recive, data_recive_host_tup, data_length_t * sizeof(Complex), cudaMemcpyHostToDevice, stream_copy);
        cudaMemcpyAsync(data_recive + data_length_t, data_recive_host_tdown, data_length_t * sizeof(Complex), cudaMemcpyHostToDevice, stream_copy);
        cudaStreamSynchronize(stream_copy);//等待内部dslash完成
        dslash_tborder_rec <<<Lz * Ly * Lx/ BLOCK_SIZE, BLOCK_SIZE , 0, stream_compute>>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_recive);
        //y
        cudaMemcpyAsync(data_recive + (Lz*Ly*Lx + Lt*Ly*Lx )*12, data_recive_host_yup, data_length_y * sizeof(Complex), cudaMemcpyHostToDevice, stream_copy);
        cudaMemcpyAsync(data_recive + (Lz*Ly*Lx + Lt*Ly*Lx )*12 + data_length_y, data_recive_host_ydown, data_length_y * sizeof(Complex), cudaMemcpyHostToDevice, stream_copy);
        cudaStreamSynchronize(stream_copy);//等待内部dslash完成
        dslash_yborder_rec <<<Lz * Lx * Lt/ BLOCK_SIZE, BLOCK_SIZE , 0, stream_compute>>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_recive);
        //z
        cudaMemcpyAsync(data_recive + (Lz*Ly*Lx)*12, data_recive_host_zup, data_length_z * sizeof(Complex), cudaMemcpyHostToDevice, stream_copy);
        cudaMemcpyAsync(data_recive + (Lz*Ly*Lx)*12 + data_length_z, data_recive_host_zdown, data_length_z * sizeof(Complex), cudaMemcpyHostToDevice, stream_copy);
        cudaStreamSynchronize(stream_copy);//等待内部dslash完成
        dslash_zborder_rec <<<Lx * Ly * Lt/ BLOCK_SIZE, BLOCK_SIZE , 0, stream_compute>>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_recive);
        //边界
        cudaStreamSynchronize(stream_compute);//等待内部dslash完成
        dslash_side_revc <<<Lx * Ly * Lz *Lt / BLOCK_SIZE, BLOCK_SIZE >>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_recive,1);
        cudaDeviceSynchronize();
        //求和


    }

    /*
    dslash乘法
    功能：
    U矩阵与fermi_in向量的乘法，结果放在fermi_out中
    */
    void dslash_multiply_nccl(Complex *gauge, Complex *fermi_out, Complex *fermi_in, bool parity_input){
        if (if_init == 0)
        {
            printf("ERROR: dslash_gpu还未初始化!");
            exit(0);
        }

        //参数输入
        parity = parity_input;
        d_gauge = gauge;
        d_fermi_in = fermi_in;
        d_fermi_out = fermi_out;

        //计算需要传输的边界
        dslash_tborder <<<Lx * Ly * Lz  / BLOCK_SIZE, BLOCK_SIZE, 0, stream_copy >>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_send);//计算t方向的边界
        dslash_xborder <<<Lt * Ly * Lz  / BLOCK_SIZE, BLOCK_SIZE, 0, stream_copy >>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_send + (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx)*12);//计算x方向的边界 
        dslash_zborder <<<Lt * Ly * Lx  / BLOCK_SIZE, BLOCK_SIZE, 0, stream_copy >>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_send + (Lz*Ly*Lx)*12);//计算z方向的边界
        dslash_yborder <<<Lt * Lz * Lx  / BLOCK_SIZE, BLOCK_SIZE, 0, stream_copy >>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_send + (Lz*Ly*Lx + Lt*Ly*Lx)*12);//计算y方向的边界

        dslash_inner <<<Lx * Ly  * (Lz ) * (Lt)  / BLOCK_SIZE, BLOCK_SIZE , 0, stream_compute>>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity);

        //等待内部dslash完成
        cudaStreamSynchronize(stream_copy);

        //mpi开始传输(cpu->cpu)
        int add;

        //t_up
        ncclGroupStart();
        ncclSend(data_send,   data_length_t * 2, ncclDouble, (my_rank + 1 - (t == (gpu_assignment[3]-1))*gpu_assignment[3]), comm, stream_copy);
        ncclRecv(data_recive, data_length_t * 2, ncclDouble, (my_rank - 1 + (t == 0)*gpu_assignment[3])                    , comm, stream_copy);

        //t_down
        ncclSend(data_send + data_length_t,   data_length_t * 2, ncclDouble, (my_rank - 1 + (t == 0)*gpu_assignment[3])                    , comm, stream_copy);
        ncclRecv(data_recive + data_length_t, data_length_t * 2, ncclDouble, (my_rank + 1 - (t == (gpu_assignment[3]-1))*gpu_assignment[3]), comm, stream_copy);
        

        //x_up
        add = gpu_assignment[3] * gpu_assignment[2] * gpu_assignment[1];
        my_size = gpu_assignment[3] * gpu_assignment[2] * gpu_assignment[1] * gpu_assignment[0];
        ncclSend(data_send + (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx)*12,   data_length_x * 2, ncclDouble, (my_rank + add - (x == (gpu_assignment[0]-1))*my_size), comm, stream_copy);
        ncclRecv(data_recive + (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx)*12, data_length_x * 2, ncclDouble, (my_rank - add + (x == 0)*my_size)                    , comm, stream_copy);

        //x_down
        ncclSend(data_send + (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx)*12 + data_length_x,   data_length_x * 2, ncclDouble, (my_rank - add + (x == 0)*my_size)                    , comm, stream_copy);
        ncclRecv(data_recive + (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx)*12 + data_length_x, data_length_x * 2, ncclDouble, (my_rank + add - (x == (gpu_assignment[0]-1))*my_size), comm, stream_copy);
        
        //y_up
        add = gpu_assignment[3] * gpu_assignment[2];
        my_size = gpu_assignment[3] * gpu_assignment[2] * gpu_assignment[1];
        ncclSend(data_send + (Lz*Ly*Lx + Lt*Ly*Lx )*12,   data_length_y * 2, ncclDouble, (my_rank + add - (y == (gpu_assignment[1]-1))*my_size), comm, stream_copy);
        ncclRecv(data_recive + (Lz*Ly*Lx + Lt*Ly*Lx )*12, data_length_y * 2, ncclDouble, (my_rank - add + (y == 0)*my_size)                    , comm, stream_copy);

        //y_down
        ncclSend(data_send + (Lz*Ly*Lx + Lt*Ly*Lx )*12 + data_length_y,   data_length_y * 2, ncclDouble, (my_rank - add + (y == 0)*my_size)                    , comm, stream_copy);
        ncclRecv(data_recive + (Lz*Ly*Lx + Lt*Ly*Lx )*12 + data_length_y, data_length_y * 2, ncclDouble, (my_rank + add - (y == (gpu_assignment[1]-1))*my_size), comm, stream_copy);
        //z_up
        add = gpu_assignment[3];
        my_size = gpu_assignment[3] * gpu_assignment[2];
        ncclSend(data_send + (Lz*Ly*Lx)*12,   data_length_z * 2, ncclDouble, (my_rank + add - (z == (gpu_assignment[2]-1))*my_size), comm, stream_copy);
        ncclRecv(data_recive + (Lz*Ly*Lx)*12, data_length_z * 2, ncclDouble, (my_rank - add + (z == 0)*my_size)                    , comm, stream_copy);
        //z_down
        ncclSend(data_send + (Lz*Ly*Lx)*12 + data_length_z,   data_length_z * 2, ncclDouble, (my_rank - add + (z == 0)*my_size)                    , comm, stream_copy);
        ncclRecv(data_recive + (Lz*Ly*Lx)*12 + data_length_z, data_length_z * 2, ncclDouble, (my_rank + add - (z == (gpu_assignment[2]-1))*my_size), comm, stream_copy);



        
        ncclGroupEnd();
        //计算非边缘的乘法内容
        

        
        cudaStreamSynchronize(stream_compute); // 等待内部dslash完成
        cudaStreamSynchronize(stream_copy);//等待内部dslash完成
        
        //接收并计算边缘的乘法
        //x
        dslash_xborder_rec <<<Lz * Ly * Lt/ BLOCK_SIZE, BLOCK_SIZE , 0, stream_copy>>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_recive);
        //t
        dslash_tborder_rec <<<Lz * Ly * Lx/ BLOCK_SIZE, BLOCK_SIZE , 0, stream_copy>>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_recive);
        //y
        dslash_yborder_rec <<<Lz * Lx * Lt/ BLOCK_SIZE, BLOCK_SIZE , 0, stream_copy>>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_recive);
        //z
        dslash_zborder_rec <<<Lx * Ly * Lt/ BLOCK_SIZE, BLOCK_SIZE , 0, stream_copy>>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_recive);
        //边界
        cudaStreamSynchronize(stream_copy); // 等待内部dslash完成
        
        dslash_side_revc <<<Lx * Ly * Lz *Lt / BLOCK_SIZE, BLOCK_SIZE , 0, stream_copy>>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_recive,1);
        cudaDeviceSynchronize();
        //求和
    }
    
    /*
    带clover项dslash乘法
    功能：
    U矩阵与fermi_in向量的乘法，结果放在fermi_out中
    */
    void dslash_multiply_clover_nccl(Complex *gauge, Complex *fermi_out, Complex *fermi_in, bool parity_input){
        if (if_init == 0)
        {
            printf("ERROR: dslash_gpu还未初始化!");
            exit(0);
        }

        //参数输入
        parity = parity_input;
        d_gauge = gauge;
        d_fermi_in = fermi_in;
        d_fermi_out = fermi_out;

        //计算需要传输的边界
        dslash_tborder <<<Lx * Ly * Lz  / BLOCK_SIZE, BLOCK_SIZE, 0, stream_copy >>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_send);//计算t方向的边界
        dslash_xborder <<<Lt * Ly * Lz  / BLOCK_SIZE, BLOCK_SIZE, 0, stream_copy >>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_send + (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx)*12);//计算x方向的边界 
        dslash_zborder <<<Lt * Ly * Lx  / BLOCK_SIZE, BLOCK_SIZE, 0, stream_copy >>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_send + (Lz*Ly*Lx)*12);//计算z方向的边界
        dslash_yborder <<<Lt * Lz * Lx  / BLOCK_SIZE, BLOCK_SIZE, 0, stream_copy >>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_send + (Lz*Ly*Lx + Lt*Ly*Lx)*12);//计算y方向的边界

        dslash_inner <<<Lx * Ly  * (Lz ) * (Lt)  / BLOCK_SIZE, BLOCK_SIZE , 0, stream_compute>>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity);

        //等待内部dslash完成
        cudaStreamSynchronize(stream_copy);

        //mpi开始传输(cpu->cpu)
        int add;

        //t_up
        ncclGroupStart();
        ncclSend(data_send,   data_length_t * 2, ncclDouble, (my_rank + 1 - (t == (gpu_assignment[3]-1))*gpu_assignment[3]), comm, stream_copy);
        ncclRecv(data_recive, data_length_t * 2, ncclDouble, (my_rank - 1 + (t == 0)*gpu_assignment[3])                    , comm, stream_copy);

        //t_down
        ncclSend(data_send + data_length_t,   data_length_t * 2, ncclDouble, (my_rank - 1 + (t == 0)*gpu_assignment[3])                    , comm, stream_copy);
        ncclRecv(data_recive + data_length_t, data_length_t * 2, ncclDouble, (my_rank + 1 - (t == (gpu_assignment[3]-1))*gpu_assignment[3]), comm, stream_copy);
        

        //x_up
        add = gpu_assignment[3] * gpu_assignment[2] * gpu_assignment[1];
        my_size = gpu_assignment[3] * gpu_assignment[2] * gpu_assignment[1] * gpu_assignment[0];
        ncclSend(data_send + (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx)*12,   data_length_x * 2, ncclDouble, (my_rank + add - (x == (gpu_assignment[0]-1))*my_size), comm, stream_copy);
        ncclRecv(data_recive + (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx)*12, data_length_x * 2, ncclDouble, (my_rank - add + (x == 0)*my_size)                    , comm, stream_copy);

        //x_down
        ncclSend(data_send + (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx)*12 + data_length_x,   data_length_x * 2, ncclDouble, (my_rank - add + (x == 0)*my_size)                    , comm, stream_copy);
        ncclRecv(data_recive + (Lz*Ly*Lx + Lt*Ly*Lx + Lt*Lz*Lx)*12 + data_length_x, data_length_x * 2, ncclDouble, (my_rank + add - (x == (gpu_assignment[0]-1))*my_size), comm, stream_copy);
        
        //y_up
        add = gpu_assignment[3] * gpu_assignment[2];
        my_size = gpu_assignment[3] * gpu_assignment[2] * gpu_assignment[1];
        ncclSend(data_send + (Lz*Ly*Lx + Lt*Ly*Lx )*12,   data_length_y * 2, ncclDouble, (my_rank + add - (y == (gpu_assignment[1]-1))*my_size), comm, stream_copy);
        ncclRecv(data_recive + (Lz*Ly*Lx + Lt*Ly*Lx )*12, data_length_y * 2, ncclDouble, (my_rank - add + (y == 0)*my_size)                    , comm, stream_copy);

        //y_down
        ncclSend(data_send + (Lz*Ly*Lx + Lt*Ly*Lx )*12 + data_length_y,   data_length_y * 2, ncclDouble, (my_rank - add + (y == 0)*my_size)                    , comm, stream_copy);
        ncclRecv(data_recive + (Lz*Ly*Lx + Lt*Ly*Lx )*12 + data_length_y, data_length_y * 2, ncclDouble, (my_rank + add - (y == (gpu_assignment[1]-1))*my_size), comm, stream_copy);
        //z_up
        add = gpu_assignment[3];
        my_size = gpu_assignment[3] * gpu_assignment[2];
        ncclSend(data_send + (Lz*Ly*Lx)*12,   data_length_z * 2, ncclDouble, (my_rank + add - (z == (gpu_assignment[2]-1))*my_size), comm, stream_copy);
        ncclRecv(data_recive + (Lz*Ly*Lx)*12, data_length_z * 2, ncclDouble, (my_rank - add + (z == 0)*my_size)                    , comm, stream_copy);
        //z_down
        ncclSend(data_send + (Lz*Ly*Lx)*12 + data_length_z,   data_length_z * 2, ncclDouble, (my_rank - add + (z == 0)*my_size)                    , comm, stream_copy);
        ncclRecv(data_recive + (Lz*Ly*Lx)*12 + data_length_z, data_length_z * 2, ncclDouble, (my_rank + add - (z == (gpu_assignment[2]-1))*my_size), comm, stream_copy);



        
        ncclGroupEnd();
        //计算非边缘的乘法内容
        

        
        cudaStreamSynchronize(stream_compute); // 等待内部dslash完成
        cudaStreamSynchronize(stream_copy);//等待内部dslash完成
        
        //接收并计算边缘的乘法
        //x
        dslash_xborder_rec <<<Lz * Ly * Lt/ BLOCK_SIZE, BLOCK_SIZE , 0, stream_copy>>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_recive);
        //t
        dslash_tborder_rec <<<Lz * Ly * Lx/ BLOCK_SIZE, BLOCK_SIZE , 0, stream_copy>>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_recive);
        //y
        dslash_yborder_rec <<<Lz * Lx * Lt/ BLOCK_SIZE, BLOCK_SIZE , 0, stream_copy>>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_recive);
        //z
        dslash_zborder_rec <<<Lx * Ly * Lt/ BLOCK_SIZE, BLOCK_SIZE , 0, stream_copy>>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_recive);
        //边界
        cudaStreamSynchronize(stream_copy); // 等待内部dslash完成
        
        dslash_side_revc <<<Lx * Ly * Lz *Lt / BLOCK_SIZE, BLOCK_SIZE , 0, stream_copy>>> (d_gauge, d_fermi_in, d_fermi_out, Lt, Lz, Ly, Lx, parity, data_recive,1);
        cudaDeviceSynchronize();
        // 求和

        
    }
    /*
    dslash_dot
    点乘
    返回值：点乘结果的实部
    */
    void dslash_dot(Complex *d_fermi_1,Complex *d_fermi_2,Complex *dot_result){

        // dot_gpu_fermi_1<<<Lt*Lz*2 , Ly/2 >>>(d_fermi_1, d_fermi_2, dot_buf, Lt, Lz, Ly, Lx);
        // cudaDeviceSynchronize();
        // dot_gpu_fermi_2<<<1 , Lt >>>(dot_buf, Lt, Lz, Ly, Lx, dot_result);
        // cudaDeviceSynchronize();
        // // dot_result[0].real=1;

        cublasHandle_t cublasH = NULL;
        (cublasCreate(&cublasH));
        const int incx = 1;
        const int incy = 1;
        const std::vector<data_type> A = {{1.1, 1.2}, {2.3, 2.4}, {3.5, 3.6}, {4.7, 4.8}};
        const std::vector<data_type> B = {{5.1, 5.2}, {6.3, 6.4}, {7.5, 7.6}, {8.7, 8.8}};
 
        // data_type *d_A = (data_type*)(d_fermi_1);
        // data_type *d_B = (data_type *)(d_fermi_2);
        data_type *d_A = nullptr;
        data_type *d_B = nullptr;
        (cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
        (cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));


        (cublasZdotc(cublasH, 1, d_A, incx, d_B, incy, (data_type*)(&dot_result)));
        
        cublasDestroy(cublasH);


        MPI_Comm_size(MPI_COMM_WORLD, &my_size);
        //将不同gpu计算结果mpi传递至rank为0的进程进行加和
        if(my_rank == 0){
            // printf("my_size = %d\n", my_size);
            for (int i=1;i<my_size;i++){
                // printf("%d",i);
                MPI_Irecv(dot_mpi_buf + i, 2, MPI_DOUBLE, i, i, MPI_COMM_WORLD, request_dot+i);
            }
            // printf("aaaaa\n");
            //等待mpi传递完成
            for (int i=1;i<my_size;i++){
                // printf("wait\n");
                MPI_Wait(request_dot+i, &status);
                // printf("recv %d\n", i);
            }
        }
        else{
            // printf("send %d\n",my_rank);
            MPI_Isend(dot_result, 2, MPI_DOUBLE, 0, my_rank, MPI_COMM_WORLD, request_dot);
            MPI_Wait(request_dot, &status);
        }

        //求和并把结果传回所有进程
        if(my_rank == 0){
            
            for (int i=1;i<my_size;i++){
                dot_result[0].real += dot_mpi_buf[i].real;
                dot_result[0].imag += dot_mpi_buf[i].imag;
            }

            // printf("dot_result = %lf + %lfi\n",dot_result[0].real,dot_result[0].imag);

            for (int i=1;i<my_size;i++){
                MPI_Isend(dot_result, 2, MPI_DOUBLE, i, i + my_size, MPI_COMM_WORLD, request_dot+i);
            }

            //等待mpi传递完成
            for (int i=1;i<my_size;i++){
                MPI_Wait(request_dot+i, &status);
            }


        }else{
            MPI_Irecv(dot_result, 2, MPI_DOUBLE, 0, my_rank + my_size, MPI_COMM_WORLD, request_dot);
            MPI_Wait(request_dot, &status);
            
        }
        // printf("rank = %d , result = %lf\n", my_rank, dot_result[0].real);
        
    }

    /*
    dslash_eo
    两次矩阵乘法和一次加法计算
    */
    void dslash_eo(Complex *gauge, Complex *fermi_out, Complex *fermi_in, Complex *buf, Complex kappa){

        if (if_init == 0)
        {
            printf("ERROR: dslash_gpu还未初始化!");
            exit(0);
        }

        dslash_multiply_nccl(gauge, buf, fermi_in, 0);
        dslash_multiply_nccl(gauge, fermi_out, buf, 1);

        add_gpu_fermi_dslash<<< Lt*Lz*Ly*Lx/8,8>>>(fermi_out, fermi_in, kappa);
        cudaDeviceSynchronize();

        // dslash_dot(fermi_out, fermi_out, dot_result);
        // printf("host = %lf\n", dot_result[0].real);
        
    }

    /*
    清除分配的显存
    */
    void end(){
        cudaFree(data_send);
        cudaFreeHost(data_send_host_t);
        cudaFreeHost(data_send_host_y);
        cudaFreeHost(data_send_host_x);
        cudaFreeHost(data_send_host_z);
        cudaFree(data_recive);
        cudaFreeHost(data_recive_host_tup);
        cudaFreeHost(data_recive_host_tdown);
        cudaFreeHost(data_recive_host_xup);
        cudaFreeHost(data_recive_host_xdown);
        cudaFreeHost(data_recive_host_yup);
        cudaFreeHost(data_recive_host_ydown);
        cudaFreeHost(data_recive_host_zup);
        cudaFreeHost(data_recive_host_zdown);
        cudaFree(dot_buf);
        cudaFree(dot_result);
        cudaFree(dot_mpi_buf);
        cudaFreeHost(request_dot);
    }

    /*
    a-b
    */
    Complex minus_cpu(Complex a, Complex b){
        a.real -= b.real;
        a.imag -= b.imag;
        return a;
    }

    /*
    x = a/b
    */
    void divide_cpu(Complex &x, Complex a, Complex b){
        double denom = b.real * b.real + b.imag * b.imag;
        x.real = (a.real * b.real + a.imag * b.imag) / denom;
        x.imag = (a.imag * b.real - a.real * b.imag) / denom;
    }

    /*
    x = a*b
    */
    void multy_cpu(Complex &x, Complex a, Complex b){
        x.real = (a.real * b.real - a.imag * b.imag);
        x.imag = (a.imag * b.real + a.real * b.imag);
    }

    /*
    clover项
    */

};



void dslashQcu(void* fermion_out, void* fermion_in, void* gauge, QcuParam* param, int parity) {
    cudaEvent_t start, stop;
    float esp_time_gpu, time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    cudaError_t err = cudaGetLastError();
    int Lx = param->lattice_size[0] >> 1;
    int Ly = param->lattice_size[1];
    int Lz = param->lattice_size[2];
    int Lt = param->lattice_size[3];
    
    Dslash_class dslash1;
    Complex *kappa;
    Complex *zero;
    cudaMallocManaged(&zero, sizeof(Complex));
    cudaMallocManaged(&kappa, sizeof(Complex));
    zero->imag = 0, zero->real = 0;
    kappa->real = 0.125;
    //b = U * x 求x

    Complex* d_gauge = static_cast<Complex *>(gauge), *b = static_cast<Complex *>(fermion_in), *x = static_cast<Complex *>(fermion_out);

    Complex *r, *r0, *t, *s, *p, *buf0, *buf1, *v;
    cudaMalloc(&r   , Lx * Ly * Lz * Lt * Ls * Lc * sizeof(Complex));
    cudaMalloc(&r0   , Lx * Ly * Lz * Lt * Ls * Lc * sizeof(Complex));
    cudaMalloc(&t    , Lx * Ly * Lz * Lt * Ls * Lc * sizeof(Complex));
    cudaMalloc(&s  , Lx * Ly * Lz * Lt * Ls * Lc * sizeof(Complex));
    cudaMalloc(&p    , Lx * Ly * Lz * Lt * Ls * Lc * sizeof(Complex));
    cudaMalloc(&buf0 , Lx * Ly * Lz * Lt * Ls * Lc * sizeof(Complex));
    cudaMalloc(&buf1 , Lx * Ly * Lz * Lt * Ls * Lc * sizeof(Complex));
    cudaMalloc(&v   , Lx * Ly * Lz * Lt * Ls * Lc * sizeof(Complex));

    Complex *alpha, *beta, *complex_buf, *w, *rho, *rho1;
    cudaMallocManaged(&alpha, sizeof(Complex));
    cudaMallocManaged(&beta, sizeof(Complex));
    cudaMallocManaged(&complex_buf, sizeof(Complex)*2);
    cudaMallocManaged(&w, sizeof(Complex));
    cudaMallocManaged(&rho, sizeof(Complex));
    cudaMallocManaged(&rho1, sizeof(Complex));


    dslash1.init(Lt, Lz, Ly, Lx);


    output_zero<<<Lx * Ly * Lz * Lt / dslash1.BLOCK_SIZE, dslash1.BLOCK_SIZE>>>(x);
    cudaDeviceSynchronize();
    //r0 = r = b - Ax 当(x = 0)
    fermi_copy<<<Lx * Ly * Lz * Lt / dslash1.BLOCK_SIZE, dslash1.BLOCK_SIZE>>>(r, b);
    cudaDeviceSynchronize();
    fermi_copy<<<Lx * Ly * Lz * Lt / dslash1.BLOCK_SIZE, dslash1.BLOCK_SIZE>>>(p, r);
    cudaDeviceSynchronize();
    fermi_copy<<<Lx * Ly * Lz * Lt / dslash1.BLOCK_SIZE, dslash1.BLOCK_SIZE>>>(r0, b);
    cudaDeviceSynchronize();

    alpha[0].real = 1;
    w[0].real = 1;
    rho[0].real = 1;
    rho1[0].real = 1;
    
    int turns = 1000;
    
    for (int i = 0; i < turns; i++){
        cudaEventRecord(start, 0); // start

        //rho = <r0, r>
        dslash1.dslash_dot(r0, r, rho);

        // beta = (rho/rho1)*(alpha/w)
        dslash1.divide_cpu(complex_buf[0], rho[0], rho1[0]);
        dslash1.divide_cpu(complex_buf[1], alpha[0], w[0]);
        dslash1.multy_cpu(beta[0], complex_buf[0], complex_buf[1]);
        

        // p.data[1,:] = r.data[1,:] + beta*(p.data[1,:] - w*v.data[1,:])
        dslash1.divide_cpu(complex_buf[0], beta[0], w[0]);
        // printf("turns = %d r = %lf + %lfi  time_usage = %f\n", i, dslash1.dot_result[0].real, dslash1.dot_result[0].imag, esp_time_gpu);
        add_gpu_fermi_2<<<Lt * Lz * Ly * Lx / 8, 8>>>(p, r, p, v, beta[0], dslash1.minus_cpu(zero[0], complex_buf[0]));
        cudaDeviceSynchronize();err = cudaGetLastError();checkCudaErrors(err);
        
        
        //v = Ap
        dslash1.dslash_eo(d_gauge, v, p, buf0, kappa[0]);err = cudaGetLastError();checkCudaErrors(err);

        // alpha = rho/<r0, v>
        dslash1.dslash_dot(r0, v, complex_buf);
        dslash1.divide_cpu(alpha[0], rho[0], complex_buf[0]);

        // s = r - alpha*v
        add_gpu_fermi<<<Lt * Lz * Ly * Lx / 8, 8>>>(s, r, v, dslash1.minus_cpu(zero[0], alpha[0]));
        cudaDeviceSynchronize();err = cudaGetLastError();checkCudaErrors(err);
        // printf("s = %lf + %lfi\n", dslash1.dot_result[1].real, dslash1.dot_result[1].imag);

        // t = As
        dslash1.dslash_eo(d_gauge, t, s, buf0, kappa[0]);err = cudaGetLastError();checkCudaErrors(err);
        // printf("t = %lf + %lfi\n", dslash1.dot_result[1].real, dslash1.dot_result[1].imag);

        // w = <t, s>/<t, t>
        dslash1.dslash_dot(t, s, complex_buf);
        dslash1.dslash_dot(t, t, complex_buf+1);

        dslash1.multy_cpu(beta[0], complex_buf[0], complex_buf[1]);

        // x = x + alpha*p + w*s
        add_gpu_fermi_2<<<Lt * Lz * Ly * Lx / 8, 8>>>(x, x, p, s, alpha[0], w[0]);
        cudaDeviceSynchronize();err = cudaGetLastError();checkCudaErrors(err);
        
        // r = s - w*t
        add_gpu_fermi<<<Lt * Lz * Ly * Lx / 8, 8>>>(r, s, t, dslash1.minus_cpu(zero[0], w[0]));
        cudaDeviceSynchronize();err = cudaGetLastError();checkCudaErrors(err);

        rho1[0].real = rho[0].real;
        rho1[0].imag = rho[0].imag;


        cudaEventRecord(stop, 0);// stop
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&esp_time_gpu, start, stop);
        dslash1.dslash_dot(r, r, dslash1.dot_result);
        printf("turns = %d r = %lf + %lfi  time_usage = %f\n", i, dslash1.dot_result[0].real, dslash1.dot_result[0].imag, esp_time_gpu);
        time += esp_time_gpu;
        if(dslash1.dot_result[0].real < 10e-9)break;
    }
    printf("time = %f\n", time );
    

    // dslash1.dslash_eo(d_gauge, x, b, buf0, kappa[0]);
    // dslash1.dslash_multiply_nccl(d_gauge, x, b, parity);

    dslash1.end();

    cudaFree(r);
    cudaFree(r0);
    cudaFree(t);
    cudaFree(s);
    cudaFree(p);
    cudaFree(buf0);
    cudaFree(buf1);
    cudaFree(v);



}

void dslashCloverQcu(void *fermion_out, void *fermion_in, void *gauge,
                     QcuParam *param, int parity) {
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    cudaError_t err = cudaGetLastError();
    int Lx = param->lattice_size[0] >> 1;
    int Ly = param->lattice_size[1];
    int Lz = param->lattice_size[2];
    int Lt = param->lattice_size[3];
    
    Dslash_class dslash1;
    Complex *kappa;
    Complex *zero;
    cudaMallocManaged(&zero, sizeof(Complex));
    cudaMallocManaged(&kappa, sizeof(Complex));
    zero->imag = 0, zero->real = 0;
    kappa->real = 0.125;
    //b = U * x 求x

    Complex* d_gauge = static_cast<Complex *>(gauge), *b = static_cast<Complex *>(fermion_in), *x = static_cast<Complex *>(fermion_out);


    Complex *alpha, *beta, *complex_buf, *w, *rho, *rho1;
    cudaMallocManaged(&alpha, sizeof(Complex));
    cudaMallocManaged(&beta, sizeof(Complex));
    cudaMallocManaged(&complex_buf, sizeof(Complex)*2);
    cudaMallocManaged(&w, sizeof(Complex));
    cudaMallocManaged(&rho, sizeof(Complex));
    cudaMallocManaged(&rho1, sizeof(Complex));


    dslash1.init(Lt, Lz, Ly, Lx);

    // dslash1.dslash_multiply_nccl(d_gauge, x, b, parity);
    cudaDeviceSynchronize();
    dslash_clover_inner <<<Lx * Ly  * (Lz ) * (Lt)  / 8, 8 >>> (d_gauge, b, x, Lt, Lz, Ly, Lx, parity);
    cudaDeviceSynchronize();
    // dslash1.dslash_eo(d_gauge, x, b, buf0, kappa[0]);


    dslash1.end();


};
void mpiDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, QcuParam *grid){};
void mpiBistabCgQcu(void *gauge, QcuParam *param, QcuParam *grid){};
void ncclDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, QcuParam *grid){};
void ncclBistabCgQcu(void *gauge, QcuParam *param, QcuParam *grid){};