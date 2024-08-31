#!/bin/bash
bash ./make.sh
pushd ./test
mpirun -n 4 --mca btl tcp,vader,self,smcuda python ./test.mpi.bistabcg.qcu-np4.py
popd
