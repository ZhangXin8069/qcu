bash ./make.sh
pushd ./test
mpirun -n 2 python ./test.mpi.bistabcg.qcu-np2.py
popd
