bash ./make.sh
pushd ./test
rm log_*
mpirun -n 1 python ./test.clover.dslash.qcu-np1.py
popd
