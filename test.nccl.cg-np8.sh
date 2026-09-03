bash ./make.sh
pushd ./test
rm log_*
mpirun -n 8 python ./test.nccl.cg.qcu-np8.py
popd

