bash ./make.sh 
nvprof --profile-child-processes -f -o log_%h_%p.nvvp mpirun -np 2 ./test
