bash ./make.sh
pushd ./test
mpirun -n 3 python ./test.mpi.dslash.qcu-np3.py
popd
