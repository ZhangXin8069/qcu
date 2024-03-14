bash ./make.sh
pushd ./test
nvprof mpirun -n 1 python ./test.wilson.dslash.qcu.py
popd
