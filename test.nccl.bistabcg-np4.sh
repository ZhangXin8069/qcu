bash ./make.sh
pushd ./test
rm log_*
mpirun -n 4 python ./test.nccl.bistabcg.qcu-np4.py
popd

