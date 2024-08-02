bash ./make.sh
pushd ./test
rm log_*
nvprof -f -o log_%h_%p.nvvp mpirun -n 4 python ./test.mpi.cg.qcu-np4.py
popd
