bash ./make.sh
pushd ./test
nvprof --profile-child-processes -f -o log_%h_%p.nvvp mpirun -n 1 python ./test.nccl.dslash.qcu-np1.py
popd
