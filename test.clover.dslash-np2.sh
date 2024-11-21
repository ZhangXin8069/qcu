bash ./make.sh
pushd ./test
rm log_*
mpirun -n 2 python ./test.clover.dslash-np2.py
popd
