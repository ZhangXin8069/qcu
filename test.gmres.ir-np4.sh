bash ./make.sh
pushd ./test
rm log_*
mpirun -n 4 python ./test.gmres.ir-np4.py
popd
