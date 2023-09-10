cmake_minimum_required(VERSION 3.21)

project(qcu_project LANGUAGES CUDA)

# Set the desired CUDA architecture (modify this as needed)
set(SM_ARCH "sm_70")

# Other options like maxrregcount can be set similarly
set(MAXRREGCOUNT "256")

# Set CUDA architecture and other flags
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${SM_ARCH} -O3 --maxrregcount=${MAXRREGCOUNT}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${SM_ARCH} -O3")

# Manually set MPI include and library paths
set(MPI_INCLUDE_PATH "/usr/lib/x86_64-linux-gnu/openmpi/include")
set(MPI_CXX_LIBRARIES "/usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so")

# Add include directories, including MPI include directories
include_directories(${CMAKE_SOURCE_DIR}/include ${MPI_INCLUDE_PATH})

# Find MPI package
find_package(MPI REQUIRED)

# Add source files for the library
set(SOURCE_FILES
    src/cuda/qcu_cuda.cu
    src/cuda/clover_dslash.cu
    src/cuda/wilson_dslash.cu
    # Add more CUDA source files if needed
)

# Build the CUDA library
add_library(qcu SHARED ${SOURCE_FILES})

# Link the library to its dependencies, including MPI libraries
target_link_libraries(qcu
    ${MPI_CXX_LIBRARIES}
    # You can add any necessary dependencies here
)

# Set the output directory for the library
set_target_properties(qcu PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib
)