#!/bin/bash
bash ./make.sh
pushd ./test
mpirun -n 4 python ./test.mpi.bistabcg.qcu-np4.py
popd
