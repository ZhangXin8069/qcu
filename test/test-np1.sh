bash ./make.sh 
rm log_*
mpirun -np 1 ./test
# ncu --set all -f -o ./log_%h_%p mpirun -np 1 ./test