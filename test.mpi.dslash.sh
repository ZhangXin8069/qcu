bash ./make.sh
pushd ./test
mpirun -np 4 python ./test.mpi.dslash.qcu.py
popd
