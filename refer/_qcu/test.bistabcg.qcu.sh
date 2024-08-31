#!/bin/bash
bash ./make.sh
pushd ./test
mpirun -n 1 python ./test.bistabcg.qcu.py
popd
