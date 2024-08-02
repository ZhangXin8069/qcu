bash ./make.sh
pushd ./test
rm log_*
nvprof -f -o log_%h_%p.nvvp mpirun -n 2 python ./test.mpi.cg.qcu-np2.py
popd
