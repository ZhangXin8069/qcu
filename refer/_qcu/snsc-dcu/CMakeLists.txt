# Set the base
cmake_minimum_required(VERSION 3.8)
project(QCU LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)

# Add include directories
include_directories(../include)
# Set your source files
aux_source_directory(../src/cuda SOURCE_FILE)

find_package(MPI REQUIRED)
target_link_libraries(qcu ${MPI_LIBRARIES})
target_link_libraries(qcu ${CUDA_LIBRARIES})
#target_link_libraries(qcu nccl)
target_compile_definitions(qcu PRIVATE MPI_ENABLED)