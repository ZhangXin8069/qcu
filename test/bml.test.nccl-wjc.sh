nvcc -o test.nccl-wjc test.nccl-wjc.cu -arch=sm_70 -lnccl -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include

nvprof --profile-child-processes -f -o log_%h_%p.nvvp mpirun -np 1 test.nccl-wjc