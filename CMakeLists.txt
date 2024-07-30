# Set the base
cmake_minimum_required(VERSION 3.8)
project(QCU LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)

# Set the desired CUDA architecture (modify this as needed)
# set(SM_ARCH "sm_80") # 4060 A100
# set(SM_ARCH "sm_70") # V100
# set(SM_ARCH "sm_60") # P100
set(SM_ARCH "sm_80")
set(CMAKE_CUDA_ARCHITECTURES "80")

# Set CUDA architecture and other flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${SM_ARCH} -O3")

# Add include directories, including MPI include directories, including qcu's include
set(MPI_INCLUDE_PATH "/usr/lib/x86_64-linux-gnu/openmpi/include")
set(MPI_CXX_LIBRARIES "/usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so")
include_directories(${MPI_INCLUDE_PATH})
include_directories(include)

aux_source_directory(src SOURCE_FILE)
add_library(qcu SHARED ${SOURCE_FILE})

find_package(MPI REQUIRED)
target_link_libraries(qcu ${MPI_LIBRARIES})
target_link_libraries(qcu ${CUDA_LIBRARIES})
target_link_libraries(qcu nccl)