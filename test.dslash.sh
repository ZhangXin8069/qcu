bash ./make.sh
pushd ./test
mpirun -np 1 python ./test.dslash.qcu.py
popd
