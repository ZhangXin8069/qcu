bash ./make.sh 
nvprof -f -o log.nvvp%p mpirun -np 2 ./test
