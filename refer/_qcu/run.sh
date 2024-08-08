bash ./make.sh

# mpirun -n 2 python mpi_dslash.py
# rm test_cg.nvvp
# nvprof -o test_cg.nvvp python test_cg.py
# mpirun -n 2 python mpi_dslash.py
rm aa_*.nvvp
# mpirun -n 2 python test_cg.py
nvprof --profile-child-processes -o aa_%p.nvvp mpirun -n 2 python test_cg.py

# python test_clover.py