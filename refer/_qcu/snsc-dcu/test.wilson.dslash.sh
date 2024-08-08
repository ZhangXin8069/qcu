bash ./make.sh
pushd ./test
mpirun -n 1 --mca btl tcp,vader,self,smcuda python ./test.wilson.dslash.qcu.py
popd
