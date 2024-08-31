#!/bin/bash
bash ./make.sh
pushd ./test
mpirun -n 2 python ./test.nccl.dslash.qcu-np2.py
popd
