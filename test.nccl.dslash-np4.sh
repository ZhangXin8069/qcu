bash ./make.sh
pushd ./test
nvprof --profile-child-processes -f -o log_%h_%p.nvvp mpirun -n 4 python ./test.nccl.dslash.qcu-np4.py
popd