bash ./make.sh
pushd ./test
rm log_*
nsys profile --trace=cuda,nvtx,osrt,mpi --output=log_%h_%p mpirun -n 4 python ./test.mpi.cg.qcu-np4.py
popd
