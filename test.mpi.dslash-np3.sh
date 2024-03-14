bash ./make.sh
pushd ./test
nsys nvprof mpirun -n 3 python ./test.mpi.dslash.qcu-np3.py
popd
