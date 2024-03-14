bash ./make.sh
pushd test
mpirun -n 1 python ./test.clover.dslash.qcu.py
popd