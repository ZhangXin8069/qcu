bash ./make.sh
pushd ./test
nsys nvprof mpirun -n 8 python ./test.mpi.dslash.qcu-np8.py
popd
