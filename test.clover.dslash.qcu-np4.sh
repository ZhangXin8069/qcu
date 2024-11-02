bash ./make.sh
pushd ./test
rm log_*
nvprof --profile-child-processes --openacc-profiling off -f -o log_%h_%p.nvvp  mpirun -n 4 python ./test.clover.dslash.qcu-np4.py
popd
