# Set the base
cmake_minimum_required(VERSION 3.8)
project(QCU LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)

# Set the desired CUDA architecture (modify this as needed)
set(SM_ARCH "sm_80")

# Other options like maxrregcount can be set similarly
set(MAXRREGCOUNT "256")

# Set CUDA architecture and other flags
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${SM_ARCH} -O3 --maxrregcount=${MAXRREGCOUNT}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${SM_ARCH} -O3")

# Manually set MPI include and library paths
set(MPI_INCLUDE_PATH "/usr/lib/x86_64-linux-gnu/openmpi/include")
set(MPI_CXX_LIBRARIES "/usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so")

# Add include directories, including MPI include directories, including qcu's include
include_directories(${CMAKE_SOURCE_DIR}/include ${MPI_INCLUDE_PATH})
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



