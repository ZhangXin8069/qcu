bash ./make.sh
pushd ./test
mpirun -n 1 python ./test.nccl.dslash.qcu-np1.py
popd
