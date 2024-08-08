bash ./make.sh
pushd ./test
rm log_*
mpirun -n 2 python ./test.mpi.dslash.qcu-np2.py
popd
