bash ./make.sh
pushd ./test
mpirun -n 3 --mca btl tcp,vader,self,smcuda python ./test.mpi.dslash.qcu-np3.py
popd
