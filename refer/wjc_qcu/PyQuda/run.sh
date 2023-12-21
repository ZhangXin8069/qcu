
#source /public/home/tangd/DCU/env.sh
module purge
module load compiler/devtoolset/7.3.1
#module load hpcx/2.4.1-gcc-7.3.1
module load compiler/gcc/7.3.1
module load compiler/dtk-23.04
module load hpcx/gcc-7.3.1
#source /public/home/tangd/DCU/dtk-23.04/env.sh
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=/public/sugon/software/compiler/dtk-23.04
#source /work/share/sugonhpctest02/tangxd/env_conda.sh
#source /work/share/sugonhpctest02/tangxd/dtk-23.04/env.sh
export HIPCC=hipcc

export CC=hipcc
export CXX=hipcc
export HCC_AMDGPU_TARGET=gfx906

export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
export OMPI_MCA_opal_cuda_support=0
#export LD_LIBRARY_PATH=${INSTALLROOT}/chroma/lib:${INSTALLROOT}/quda/lib:${INSTALLROOT}/qdpxx/lib:${INSTALLROOT}/qmp/lib:${LD_LIBRARY_PATH}

#./chroma -i tests/test.clover.ini.xml
#pushd PyQuda
#python3  tests/test.clover.py
#python3 tests/test.clover.py
#mpirun -n 1  python3  tests/test.dslash.py
#mpirun -n 1  python3  tests/test.dslash.mpi.py
#mpirun -n 1  python3  tests/test.dslash.mpi.py

mpirun -n 1 python tests/test.new_storage.py
#mpirun -n 1 python tests/test.nondebug_cg.py
#test.new_storage.py
#mpirun -n 1  python3  tests/test.debug.py
#python3  tests/test.debug.py
#mpirun -n 1  python3  tests/test.dslash.new_mpi.py
#mpirun -n 2   python3  tests/test.dslash.py
#popd


