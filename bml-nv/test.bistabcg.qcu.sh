bash ./make.sh
pushd ./test
nvprof mpirun -n 1 python ./test.bistabcg.qcu.py
popd
