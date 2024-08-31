#!/bin/bash
bash ./make.sh
pushd ./test
rm log_*
nvprof --profile-child-processes -f -o log_%h_%p.nvvp mpirun -n 4 python ./test.nccl.dslash.qcu-np4.py
popd
