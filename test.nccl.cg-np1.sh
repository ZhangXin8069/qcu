bash ./make.sh
pushd ./test
rm log_*
mpirun -n 1 python ./test.nccl.cg.qcu-np1.py
popd

