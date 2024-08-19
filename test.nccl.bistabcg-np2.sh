bash ./make.sh
pushd ./test
rm log_*
mpirun -n 2 python ./test.nccl.bistabcg.qcu-np2.py
popd
rm libqcu.so
