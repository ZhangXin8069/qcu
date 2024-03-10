bash ./make.sh
pushd ../test
mpirun -n 2 --mca btl tcp,vader,self,smcuda python ./test.mpi.dslash.qcu-np2.py
popd
