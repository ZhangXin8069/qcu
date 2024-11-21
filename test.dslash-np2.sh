bash ./make.sh
pushd ./test
rm log_*
mpirun -n 2 python ./test.dslash-np2.py
popd
