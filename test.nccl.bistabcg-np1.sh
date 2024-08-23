bash ./make.sh
pushd ./test
rm log_*
mpirun -n 1 python ./test.nccl.bistabcg.qcu-np1.py
popd

