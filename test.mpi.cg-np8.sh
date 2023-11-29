bash ./make.sh
pushd ./test
mpirun -n 8 --mca btl tcp,vader,self,smcuda python ./test.mpi.cg.qcu-np8.py
popd
