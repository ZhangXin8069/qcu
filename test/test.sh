bash ./make.sh && mpirun -n 8 --mca btl tcp,vader,self,smcuda ./test && hipprof ./test
