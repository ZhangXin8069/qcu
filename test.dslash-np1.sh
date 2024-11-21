bash ./make.sh
pushd ./test
rm log_*
mpirun -n 1 python ./test.dslash-np1.py
popd
