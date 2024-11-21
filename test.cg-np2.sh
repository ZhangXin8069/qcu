bash ./make.sh
pushd ./test
rm log_*
mpirun -n 2 python ./test.cg-np2.py
popd

