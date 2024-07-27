bash ./make.sh
pushd ./test
nvprof --profile-child-processes -f -o log.nvvp%p mpirun -n 2 python ./test.mpi.bistabcg.qcu-np2.py
popd
