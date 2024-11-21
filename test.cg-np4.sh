bash ./make.sh
pushd ./test
rm log_*
mpirun -n 4 python ./test.cg-np4.py
popd
