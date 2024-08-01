bash ./make.sh
pushd ./test
rm log_*
rm log_*
nsys nvprof --profile-child-processes -f -o log_%h_%p.nvvp mpirun -n 1 python ./test.bistabcg.qcu.py
popd
