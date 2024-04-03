bash ./make.sh
pushd ./test
mpirun -n 8 python ./test.mpi.dslash.qcu-np8.py
popd
