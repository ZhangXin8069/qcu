bash ./make.sh
pushd ./test
mpirun -n 4 python ./test.nccl.bistabcg.qcu-np4.py
popd
