bash ./make.sh
pushd ./test
rm log_*
nvprof --profile-child-processes -f -o log.nvvp%p mpirun -n 1 python ./test.bistabcg.qcu.py
popd
