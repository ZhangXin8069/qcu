bash ./make.sh
pushd ./test
rm log_*
mpirun -n 4 python ./test.clover.dslash.qcu-np4.py
popd
