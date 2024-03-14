bash ./make.sh
pushd ./test
nvprof mpirun -n 2 python ./test.mpi.dslash.qcu-np2.py
popd
