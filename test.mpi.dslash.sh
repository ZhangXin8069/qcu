bash ./make.sh
pushd ./test
mpirun -np 1 python ./test.mpi.dslash.qcu.py
popd
