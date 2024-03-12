bash ./make.sh
pushd ./test
nsys nvprof mpirun -n 1 python ./test.dslash.qcu.py
popd
