cmake .
source ./env.sh
mpirun -np 1 python ./test/test.dslash.qcu.py
