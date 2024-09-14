bash ./make.sh
pushd ./test
rm log_*
mpirun -n 2 python ./test.clover.dslash.qcu-np2.py
# nsys profile -f true -o log_%h_%p mpirun -n 2 python ./test.clover.dslash.qcu-np2.py
popd
