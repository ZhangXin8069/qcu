bash ./make.sh
pushd ./test
rm log_*
nsys profile --trace=cuda,nvtx,osrt,mpi --output=log_%h_%p mpirun -n 2 python ./test.mpi.cg.qcu-np2.py
popd
