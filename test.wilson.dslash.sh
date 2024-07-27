bash ./make.sh
pushd ./test
nvprof --profile-child-processes -f -o log.nvvp%p mpirun -n 1 python ./test.wilson.dslash.qcu.py
popd
