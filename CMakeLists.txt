cmake_minimum_required(VERSION 3.8)
#include_directories(/usr/lib/x86_64-linux-gnu/openmpi/include)
#set(CMAKE_CUDA_ARCHITECTURES 70)
project(QCU LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

include_directories(include)
add_library(qcu SHARED 
src/mpi_clover_bistabcg.cu
src/mpi_clover_dslash.cu
src/mpi_clover_multgrid.cu
src/mpi_overlap_bistabcg.cu
src/mpi_overlap_dslash.cu
src/mpi_overlap_multgrid.cu
src/mpi_wilson_bistabcg.cu
src/mpi_wilson_dslash.cu
src/mpi_wilson_multgrid.cu
src/test_bistabcg.cu
src/test_dslash.cu
src/test_multgrid.cu
src/test.cu
)

find_package(MPI REQUIRED)
target_link_libraries(qcu ${MPI_LIBRARIES})
target_compile_definitions(qcu PRIVATE MPI_ENABLED)

