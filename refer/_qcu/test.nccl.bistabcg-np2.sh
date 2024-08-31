#!/bin/bash
bash ./make.sh
pushd ./test
mpirun -n 2 python ./test.nccl.bistabcg.qcu-np2.py
popd
