cmake_minimum_required(VERSION 3.21)

project(qcu_project LANGUAGES CUDA)

# Read the value of sm_arch and maxrregcount from env.sh
execute_process(
    COMMAND bash -c "source ${CMAKE_SOURCE_DIR}/env.sh >>log.txt 2>&1 && echo $sm_arch"
    OUTPUT_VARIABLE SM_ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND bash -c "source ${CMAKE_SOURCE_DIR}/env.sh >>log.txt 2>&1 && echo $maxrregcount"
    OUTPUT_VARIABLE MAXRREGCOUNT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set CUDA architecture and other flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${SM_ARCH}  -O3 --maxrregcount=${MAXRREGCOUNT}")

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add source files for the library
set(SOURCE_FILES
    src/cuda/qcu_cuda.cu
    src/cuda/clover_dslash.cu
    src/cuda/wilson_dslash.cu
    # Add more CUDA source files if needed
)

# Build the CUDA library
add_library(qcu SHARED ${SOURCE_FILES})

# Link the library to its dependencies
target_link_libraries(qcu
    # You can add any necessary dependencies here
)

# Set the output directory for the library
set_target_properties(qcu PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib
)
