cmake_minimum_required(VERSION 3.8)
project(QCU LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

include_directories(include)
add_library(qcu SHARED 
src/cuda/mpi_clover_bistabcg.cu
src/cuda/mpi_clover_dslash.cu
src/cuda/mpi_clover_multgrid.cu
src/cuda/mpi_overlap_bistabcg.cu
src/cuda/mpi_overlap_dslash.cu
src/cuda/mpi_overlap_multgrid.cu
src/cuda/mpi_wilson_bistabcg.cu
src/cuda/mpi_wilson_dslash.cu
src/cuda/mpi_wilson_multgrid.cu
src/cuda/test_clover_bistabcg.cu
src/cuda/test_clover_dslash.cu
src/cuda/test_clover_multgrid.cu
src/cuda/test_overlap_bistabcg.cu
src/cuda/test_overlap_dslash.cu
src/cuda/test_overlap_multgrid.cu
src/cuda/test_wilson_bistabcg.cu
src/cuda/test_wilson_dslash.cu
src/cuda/test_wilson_multgrid.cu
src/cuda/clover_bistabcg.cu
src/cuda/clover_dslash.cu
src/cuda/clover_multgrid.cu
src/cuda/overlap_bistabcg.cu
src/cuda/overlap_dslash.cu
src/cuda/overlap_multgrid.cu
src/cuda/wilson_bistabcg.cu
src/cuda/wilson_dslash.cu
src/cuda/wilson_multgrid.cu
)

find_package(MPI REQUIRED)
target_link_libraries(qcu ${MPI_LIBRARIES})
target_link_libraries(qcu ${CUDA_LIBRARIES})
target_compile_definitions(qcu PRIVATE MPI_ENABLED)



