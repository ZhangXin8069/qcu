set(CMAKE_CUDA_COMPILER "/public/sugon/software/compiler/dtk-23.04/cuda/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/public/sugon/software/compiler/dtk-23.04/llvm/bin/clang++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "10.2.89")
set(CMAKE_CUDA_DEVICE_LINKER "/public/sugon/software/compiler/dtk-23.04/cuda/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/public/sugon/software/compiler/dtk-23.04/cuda/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "OFF")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "")
set(CMAKE_CUDA20_COMPILE_FEATURES "")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "Clang")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "14.0")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/public/sugon/software/compiler/dtk-23.04/cuda")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/public/sugon/software/compiler/dtk-23.04/cuda")
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "10.2.89")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/public/sugon/software/compiler/dtk-23.04/cuda")

set(CMAKE_CUDA_ARCHITECTURES_ALL "30;35;37;50;52;53;60;61;62")
set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "30;35;50;60")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/public/sugon/software/compiler/dtk-23.04/cuda/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/public/sugon/software/compiler/dtk-23.04/cuda/targets/x86_64-linux/lib/stubs;/public/sugon/software/compiler/dtk-23.04/cuda/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/ompi/include;/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/ucx-without-dtk/include;/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/sharp/include;/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/hcoll/include;/public/sugon/software/compiler/dtk-23.04/llvm/lib/clang/14.0.0;/public/sugon/software/compiler/dtk-23.04/llvm/lib/clang/14.0.0/include/cuda_wrappers;/public/sugon/software/compiler/dtk-23.04/hsa/include;/public/sugon/software/compiler/dtk-23.04/cuda/include;/opt/rh/devtoolset-7/root/usr/include/c++/7;/opt/rh/devtoolset-7/root/usr/include/c++/7/x86_64-redhat-linux;/public/sugon/software/compiler/dtk-23.04/include;/public/sugon/software/compiler/dtk-23.04/llvm/include;/public/sugon/software/compiler/dtk-23.04/hip/include;/public/sugon/software/compiler/dtk-23.04/miopen/include;/opt/rh/devtoolset-7/root/usr/include/c++/7/backward;/public/sugon/software/compiler/dtk-23.04/llvm/lib/clang/14.0.0/include;/usr/local/include;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/public/sugon/software/compiler/dtk-23.04/cuda/targets/x86_64-linux/lib/stubs;/public/sugon/software/compiler/dtk-23.04/cuda/targets/x86_64-linux/lib;/opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7;/opt/rh/devtoolset-7/root/usr/lib64;/lib64;/usr/lib64;/public/sugon/software/compiler/dtk-23.04/llvm/lib;/lib;/usr/lib;/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/ompi/lib;/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/sharp/lib;/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/hcoll/lib;/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/ucx-without-dtk/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "SHARED")

set(CMAKE_LINKER "/opt/rh/devtoolset-7/root/usr/bin/ld")
set(CMAKE_AR "/opt/rh/devtoolset-7/root/usr/bin/ar")
set(CMAKE_MT "")
