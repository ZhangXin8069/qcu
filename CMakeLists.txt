cmake_minimum_required(VERSION 3.5)

set(CMAKE_CUDA_ARCHITECTURES 70)
project(QCU LANGUAGES C CXX CUDA)

include_directories(./include)
add_library(qcu SHARED src/dslash.cu)
