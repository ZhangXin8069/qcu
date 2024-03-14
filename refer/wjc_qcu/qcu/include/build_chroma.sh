#! /bin/bash

module purge
module load compiler/cmake/3.24.1
module load mpi/hpcx/2.11.0/intel-2017.5.239
module load compiler/rocm/dtk/23.04
export HIP_PLATFORM='amd'
export HIP_CLANG_PATH=$ROCM_PATH/llvm/bin

ROOT=$(pwd)
echo ${ROOT}
SOURCE=${ROOT}/source
BUILD=${ROOT}/build
INSTALL=${ROOT}/install

USQCD="/public/home/jiangxiangyu/chroma_quda/usqcd/chroma_quda_jit"
#USQCD="/public/home/ybyang/new_quda/usqcd"

export CC=clang
export CXX=clang++
export CFLAGS="-fopenmp -D_REENTRANT -g -ffast-math -funroll-loops -fomit-frame-pointer -ftree-vectorize -fassociative-math"                                    
export CXXFLAGS="-fopenmp -D_REENTRANT -g -ffast-math -funroll-loops -fomit-frame-pointer -ftree-vectorize -fassociative-math"

#source "${ROOT}/env.sh"
#source $USQCD/../src-jit/env.sh

mkdir -p ${BUILD}
cd ${BUILD}

# Remove # of the following line and execute it once, if the above paths are changed.
#rm CMakeCache.txt
# Remove # of the following line and execute it once, if more source files are added. 
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_PREFIX_PATH=${USQCD} -DCMAKE_INSTALL_PREFIX=${INSTALL} ${SOURCE}
 34 
make -j
make install