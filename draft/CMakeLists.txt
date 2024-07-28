# Set the base
cmake_minimum_required(VERSION 3.8)
project(test LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)

# Set the desired CUDA architecture (modify this as needed)
set(SM_ARCH "sm_60")

# Other options like maxrregcount can be set similarly
set(MAXRREGCOUNT "256")

# Set CUDA architecture and other flags
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${SM_ARCH} -O3 --maxrregcount=${MAXRREGCOUNT}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${SM_ARCH} -O3")
set(CMAKE_CUDA_ARCHITECTURES "60")

# Add include directories, including MPI include directories, including test's include
set(MPI_INCLUDE_PATH "/usr/lib/x86_64-linux-gnu/openmpi/include")
set(MPI_CXX_LIBRARIES "/usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so")
set(QUDA_HOME "/home/kfutfd/quda-develop")
set(QUDA_LIBRARIES ${QUDA_HOME}"/build/lib/libquda.so")
include_directories(${CMAKE_SOURCE_DIR}/include ${MPI_INCLUDE_PATH})
include_directories(../include)
include_directories(../refer/eigen)
include_directories(${QUDA_HOME}"/include")
include_directories(${QUDA_HOME}"/include/externals")
include_directories(${QUDA_HOME}"/tests")
include_directories(${QUDA_HOME}"/tests/googletest/include")
include_directories(${QUDA_HOME}"/tests/host_reference") 
include_directories(${QUDA_HOME}"/tests/utils")

include_directories(${QUDA_HOME}"/build/include")
include_directories(${QUDA_HOME}"/build/include/externals")
include_directories(${QUDA_HOME}"/build/tests")
include_directories(${QUDA_HOME}"/build/tests/googletest/include")
include_directories(${QUDA_HOME}"/build/tests/host_reference") 
include_directories(${QUDA_HOME}"/build/tests/utils")

# Set your source files
aux_source_directory(../src/cuda SOURCE_FILE)

# Add an executable and link the necessary libraries
find_package(MPI REQUIRED)
#add_executable(test test.mpi.bistabcg.cu ${SOURCE_FILE})
# add_executable(test test.nccl.bistabcg.cu ${SOURCE_FILE})
add_executable(test test.quda.cu ${SOURCE_FILE})
target_link_libraries(test Eigen)
target_link_libraries(test ${MPI_C_LIBRARIES})
target_link_libraries(test ${CUDA_LIBRARIES})
target_link_libraries(test ${QUDA_LIBRARIES})
target_link_libraries(test nccl)