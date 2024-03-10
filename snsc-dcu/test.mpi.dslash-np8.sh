bash ./make.sh
pushd ../test
mpirun -n 8 --mca btl tcp,vader,self,smcuda python ./test.mpi.dslash.qcu-np8.py
popd
