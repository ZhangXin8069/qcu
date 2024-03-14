bash ./make.sh
pushd ./test
nvprof mpirun -n 4 python ./test.mpi.bistabcg.qcu-np4.py
popd
