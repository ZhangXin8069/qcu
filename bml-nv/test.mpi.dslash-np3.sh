bash ./make.sh
pushd ./test
nvprof mpirun -n 3 python ./test.mpi.dslash.qcu-np3.py
popd
