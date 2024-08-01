bash ./make.sh 
nsys nvprof --profile-child-processes -f -o log.nvvp%p mpirun -np 1 ./test
