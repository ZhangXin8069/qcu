bash ./make.sh
pushd ./test
rm log_*
nvprof -f -o log_%h_%p.nvvp mpirun -n 1 python ./test.clover.dslash.qcu.py
popd