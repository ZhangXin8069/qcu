bash ./make.sh
pushd ./test
mpirun -n 1 python ./test.dslash.qcu.py
popd
