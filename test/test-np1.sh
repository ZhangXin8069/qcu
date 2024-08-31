#!/bin/bash
bash ./make.sh 
rm log_*
mpirun -np 1 ./test
