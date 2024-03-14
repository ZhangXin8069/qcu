bash ./make.sh
pushd ./test
mpirun -n 4 python ./test.mpi.cg.qcu-np4.py
popd
