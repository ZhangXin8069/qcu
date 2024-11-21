bash ./make.sh
pushd ./test
rm log_*
mpirun -n 4 python ./test.dslash-np4.py
popd

