bash ./make.sh
pushd ./test
rm log_*
mpirun -n 2 python ./test.gmres.ir-np2.py
popd
