bash ./make.sh
pushd ./test
mpirun -n 1 python ./test.wilson.dslash.qcu.py
popd
