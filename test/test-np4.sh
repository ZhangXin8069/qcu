bash ./make.sh 
rm log_*
nsys nvprof --profile-child-processes -f -o log_%h_%p.nvvp mpirun -np 4 ./test
