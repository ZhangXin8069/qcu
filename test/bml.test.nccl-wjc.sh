nvcc -o test.nccl-wjc test.nccl-wjc.cu -arch=sm_70 -lnccl -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include

mpirun -np 4 test.nccl-wjc