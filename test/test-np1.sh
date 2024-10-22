bash ./make.sh 
rm log_*
# ncu --set all -f -o ./log_%h_%p nvprof -f -o log_%h_%p.nvvp  mpirun -np 1 ./test
mpirun -np 1 ./test