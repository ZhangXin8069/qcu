bash ./make.sh
pushd ./test
mpirun -n 1 python ./test.mpi.cg.qcu-np1.py
popd
