cmake_minimum_required(VERSION 3.8)
include_directories(/usr/lib/x86_64-linux-gnu/openmpi/include)
#set(CMAKE_CUDA_ARCHITECTURES 906)
project(QCU LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
#add_definitions("-g")


include_directories(include)
#add_library(qcu SHARED src/dslash.cu)
add_library(qcu SHARED src/qcu_complex_computation.cu src/cg_inverter.cu  src/qcu_shift_storage.cu  src/dslash.cu src/qcu_wilson_dslash.cu  src/qcu_clover_dslash.cu  src/qcu_wilson_dslash_neo.cu  src/qcu_communicator.cu)
#target_compile_options(qcu PRIVATE "-Wall -Wextra")

find_package(MPI REQUIRED)
target_link_libraries(qcu ${MPI_LIBRARIES})
target_compile_definitions(qcu PRIVATE MPI_ENABLED)

