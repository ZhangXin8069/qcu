bash ./make.sh
pushd ./test
nsys nvprof mpirun -n 2 python ./test.mpi.dslash.qcu-np2.py
popd
