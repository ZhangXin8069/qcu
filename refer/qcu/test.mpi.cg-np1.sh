bash ./make.sh
pushd ./test
rm log_*
nsys profile --trace=cuda,nvtx,osrt,mpi --output=log_%h_%p mpirun -n 1 python ./test.mpi.cg.qcu-np1.py
popd
