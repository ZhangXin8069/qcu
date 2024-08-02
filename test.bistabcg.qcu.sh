bash ./make.sh
pushd ./test
rm log_*
rm log_*
nvprof -f -o log_%h_%p.nvvp mpirun -n 1 python ./test.bistabcg.qcu.py
popd
