bash ./make.sh
pushd ./test
rm log_*
mpirun -n 1 python ./test.gmres.ir-np1.py
popd
