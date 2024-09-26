bash ./make.sh
pushd ./test
rm log_*
nsys profile -f true -o log_%h_%p mpirun -n 1 python ./test.nccl.dslash.qcu-np1.py
popd

