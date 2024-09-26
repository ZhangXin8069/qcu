bash ./make.sh 
rm log_*
ncu --set all -f -o ./log_%h_%p mpirun -np 8 ./test
