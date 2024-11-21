bash ./make.sh
pushd ./test
rm log_*
mpirun -n 1 python ./test.cg-np1.py
popd
