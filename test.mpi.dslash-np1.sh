bash ./make.sh
pushd ./test
nsys nvprof mpirun -n 1 python ./test.mpi.dslash.qcu-np1.py
popd
