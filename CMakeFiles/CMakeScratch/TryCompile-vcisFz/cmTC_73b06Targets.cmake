# Generated by CMake

if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" LESS 2.8)
   message(FATAL_ERROR "CMake >= 2.8.0 required")
endif()
if(CMAKE_VERSION VERSION_LESS "2.8.3")
   message(FATAL_ERROR "CMake >= 2.8.3 required")
endif()
cmake_policy(PUSH)
cmake_policy(VERSION 2.8.3...3.24)
#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Create imported target MPI::MPI_CXX
add_library(MPI::MPI_CXX INTERFACE IMPORTED)

set_target_properties(MPI::MPI_CXX PROPERTIES
  INTERFACE_COMPILE_DEFINITIONS ""
  INTERFACE_COMPILE_OPTIONS "\$<\$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:SHELL:-Xcompiler >-pthread"
  INTERFACE_INCLUDE_DIRECTORIES "/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/ompi/include/openmpi;/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/ompi/include/openmpi/opal/mca/hwloc/hwloc201/hwloc/include;/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/ompi/include/openmpi/opal/mca/event/libevent2022/libevent;/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/ompi/include/openmpi/opal/mca/event/libevent2022/libevent/include;/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/ompi/include"
  INTERFACE_LINK_LIBRARIES "/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/ompi/lib/libmpi.so"
  INTERFACE_LINK_OPTIONS "\$<HOST_LINK:SHELL:-Wl\$<COMMA>-rpath -Wl\$<COMMA>/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/ompi/lib -Wl\$<COMMA>--enable-new-dtags -pthread>"
  SYSTEM "ON"
)

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
cmake_policy(POP)
