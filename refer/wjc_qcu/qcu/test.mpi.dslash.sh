bash ./make.sh
pushd ./test
mpirun -n 4 python ./test.mpi.dslash.qcu.py
popd
