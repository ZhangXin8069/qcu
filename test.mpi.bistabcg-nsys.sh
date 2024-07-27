bash ./make.sh
pushd ./test
nsys nvprof nvprof --profile-child-processes -f -o log.nvvp%p mpirun -n 1 python ./test.mpi.bistabcg.qcu-np1.py
popd
