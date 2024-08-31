#!/bin/bash
bash ./make.sh
pushd ./test
mpirun -n 1 python ./test.nccl.bistabcg.qcu-np1.py
popd
