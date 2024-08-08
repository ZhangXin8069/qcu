bash ./make.sh
pushd ./test
rm log_*
rm log_*
mpirun -n 1 python ./test.bistabcg.qcu.py
popd
