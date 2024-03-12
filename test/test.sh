bash ./make.sh && nsys nvprof mpirun -n 8 ./test && hipprof ./test
