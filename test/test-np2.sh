bash ./make.sh 
nvprof --profile-child-processes -f -o log.nvvp%p mpirun -np 2 ./test
