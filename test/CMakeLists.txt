cmake_minimum_required(VERSION 3.8)
project(test LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)


# Find MPI package
find_package(MPI REQUIRED)

# Set CUDA as required
find_package(CUDA REQUIRED)

# Include MPI and CUDA headers and other
include_directories(test ../include)
include_directories(test ${MPI_C_INCLUDE_DIRS})
include_directories(test ${CUDA_INCLUDE_DIRS})

# Set your source files
set(SOURCES
    test.cu
)

# Add an executable and link the necessary libraries
add_executable(test ${SOURCES})
target_link_libraries(test ${MPI_C_LIBRARIES})
target_link_libraries(test ${CUDA_LIBRARIES})



