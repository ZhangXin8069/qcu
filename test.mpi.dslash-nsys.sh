bash ./make.sh
pushd ./test
nsys nvprof nvprof --profile-child-processes -f -o log_%h_%p.nvvp mpirun -n 1 python ./test.mpi.dslash.qcu-np1.py
popd
