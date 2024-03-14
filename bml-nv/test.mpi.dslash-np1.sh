bash ./make.sh
pushd ./test
nvprof mpirun -n 1 python ./test.mpi.dslash.qcu-np1.py
popd
