bash ./make.sh
pushd ./test
mpirun -n 3 python ./test.mpi.cg.qcu-np3.py
popd
