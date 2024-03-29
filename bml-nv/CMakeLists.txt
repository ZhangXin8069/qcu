# Set the base
cmake_minimum_required(VERSION 3.8)
project(QCU LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)

# Set the desired CUDA architecture (modify this as needed)
set(SM_ARCH "sm_70")

# Other options like maxrregcount can be set similarly
set(MAXRREGCOUNT "256")

# Set CUDA architecture and other flags
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${SM_ARCH} -O3 --maxrregcount=${MAXRREGCOUNT}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${SM_ARCH} -O3")
set(CMAKE_CUDA_ARCHITECTURES "70")

# Manually set MPI include and library paths
set(MPI_INCLUDE_PATH "/usr/lib/x86_64-linux-gnu/openmpi/include")
set(MPI_CXX_LIBRARIES "/usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so")

# Add include directories, including MPI include directories, including qcu's include
include_directories(${CMAKE_SOURCE_DIR}/include ${MPI_INCLUDE_PATH})
include_directories(include)

aux_source_directory(../src/cuda SOURCE_FILE)
add_library(qcu SHARED ${SOURCE_FILE})

find_package(MPI REQUIRED)
target_link_libraries(qcu ${MPI_LIBRARIES})
target_link_libraries(qcu ${CUDA_LIBRARIES})
target_compile_definitions(qcu PRIVATE MPI_ENABLED)



