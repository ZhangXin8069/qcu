bash ./make.sh 
rm log_*
nvprof -f -o log_%h_%p.nvvp mpirun -np 2 ./test
