bash ./make.sh 
rm log_*
nsys profile -f true -o log_%h_%p mpirun -np 1 ./test
# mpirun -np 1 ./test