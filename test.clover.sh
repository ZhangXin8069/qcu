bash ./make.sh
pushd ./test
mpirun -np 1 python ./test.clover.dslash.qcu.py
popd
