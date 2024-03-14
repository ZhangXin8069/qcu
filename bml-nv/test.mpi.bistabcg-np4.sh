bash ./make.sh
pushd ./test
nsys nvprof mpirun -n 4 python ./test.mpi.bistabcg.qcu-np4.py
popd
