set(CMAKE_CUDA_COMPILER "/public/soft/linux-centos7-x86_64/gcc-10.3.0/cuda-11.4.4-uizl3zvwy66u3bqllanvuxwxxwfyytwo/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "11.4.152")
set(CMAKE_CUDA_DEVICE_LINKER "/public/soft/linux-centos7-x86_64/gcc-10.3.0/cuda-11.4.4-uizl3zvwy66u3bqllanvuxwxxwfyytwo/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/public/soft/linux-centos7-x86_64/gcc-10.3.0/cuda-11.4.4-uizl3zvwy66u3bqllanvuxwxxwfyytwo/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "14")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "10.3")



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

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/public/soft/linux-centos7-x86_64/gcc-10.3.0/cuda-11.4.4-uizl3zvwy66u3bqllanvuxwxxwfyytwo")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/public/soft/linux-centos7-x86_64/gcc-10.3.0/cuda-11.4.4-uizl3zvwy66u3bqllanvuxwxxwfyytwo")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/public/soft/linux-centos7-x86_64/gcc-10.3.0/cuda-11.4.4-uizl3zvwy66u3bqllanvuxwxxwfyytwo")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/public/soft/linux-centos7-x86_64/gcc-10.3.0/cuda-11.4.4-uizl3zvwy66u3bqllanvuxwxxwfyytwo/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/public/soft/linux-centos7-x86_64/gcc-10.3.0/cuda-11.4.4-uizl3zvwy66u3bqllanvuxwxxwfyytwo/targets/x86_64-linux/lib/stubs;/public/soft/linux-centos7-x86_64/gcc-10.3.0/cuda-11.4.4-uizl3zvwy66u3bqllanvuxwxxwfyytwo/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/public/soft/linux-centos7-x86_64/intel-2022.0.2/zlib-1.2.11-b7lezuxwlgr5na2wyzyuluy5ccvdsubu/include;/public/soft/linux-centos7-x86_64/intel-2022.0.2/gdbm-1.19-d6biocotphllwmxpcv73dj6i2tfg76wg/include;/public/soft/linux-centos7-x86_64/intel-2022.0.2/readline-8.1-rdbh6eyrwvgk7puiuy73bm6wfsuhpekf/include;/public/soft/linux-centos7-x86_64/intel-2022.0.2/ncurses-6.2-6bbomi7ea6lf6fbo4htoipiu6dvaxsip/include;/public/soft/linux-centos7-x86_64/intel-2022.0.2/bzip2-1.0.8-5sh6f2asi3gazgfsghv5delerfqmkpiz/include;/public/soft/linux-centos7-x86_64/intel-2022.0.2/berkeley-db-18.1.40-eybdxegfzsgulufzpjmxoq7na3xhlafw/include;/public/soft/linux-centos7-x86_64/intel-2022.0.2/libsigsegv-2.13-ptpbxgbp62qh7safgt6u75fed3xec7ge/include;/public/soft/linux-centos7-x86_64/gcc-10.3.0/openmpi-4.1.5-atnooy4j3scc3fxoqn5scmzvpjqmoa3p/include;/public/soft/linux-centos7-x86_64/gcc-10.3.0/ucx-1.14.0-thg2ethlnec5vdv6cmpok3zg52lfkvoy/include;/public/soft/linux-centos7-x86_64/gcc-10.3.0/pmix-4.2.3-ubyezqsqijflgxmbem7rnu4durhmxihh/include;/public/soft/linux-centos7-x86_64/gcc-10.3.0/libevent-2.1.12-4pbdwhqbk3r2vgv54webjc4w4pwv6dqw/include;/public/soft/linux-centos7-x86_64/gcc-10.3.0/hwloc-2.9.1-n35tnosj7xojsolcwnoamonp2zgk3xo2/include;/public/soft/linux-centos7-x86_64/gcc-10.3.0/miniconda3-4.10.3-cywznglw5bpikxtaolfbb3vvo6uq2dtx/include;/public/sugon/software/compiler/dtk-23.04/include;/public/sugon/software/compiler/dtk-23.04/llvm/include;/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/include/c++/10.3.0;/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/include/c++/10.3.0/x86_64-pc-linux-gnu;/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/include/c++/10.3.0/backward;/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/lib/gcc/x86_64-pc-linux-gnu/10.3.0/include;/usr/local/include;/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/include;/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/lib/gcc/x86_64-pc-linux-gnu/10.3.0/include-fixed;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/public/soft/linux-centos7-x86_64/gcc-10.3.0/cuda-11.4.4-uizl3zvwy66u3bqllanvuxwxxwfyytwo/targets/x86_64-linux/lib/stubs;/public/soft/linux-centos7-x86_64/gcc-10.3.0/cuda-11.4.4-uizl3zvwy66u3bqllanvuxwxxwfyytwo/targets/x86_64-linux/lib;/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/lib64;/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/lib/gcc/x86_64-pc-linux-gnu/10.3.0;/lib64;/usr/lib64;/public/soft/linux-centos7-x86_64/intel-2022.0.2/perl-5.34.0-ufjmy3aryaitjhpvw5ilo2t3auuyqtbv/lib;/public/soft/linux-centos7-x86_64/intel-2022.0.2/zlib-1.2.11-b7lezuxwlgr5na2wyzyuluy5ccvdsubu/lib;/public/soft/linux-centos7-x86_64/intel-2022.0.2/gdbm-1.19-d6biocotphllwmxpcv73dj6i2tfg76wg/lib;/public/soft/linux-centos7-x86_64/intel-2022.0.2/readline-8.1-rdbh6eyrwvgk7puiuy73bm6wfsuhpekf/lib;/public/soft/linux-centos7-x86_64/intel-2022.0.2/ncurses-6.2-6bbomi7ea6lf6fbo4htoipiu6dvaxsip/lib;/public/soft/linux-centos7-x86_64/intel-2022.0.2/bzip2-1.0.8-5sh6f2asi3gazgfsghv5delerfqmkpiz/lib;/public/soft/linux-centos7-x86_64/intel-2022.0.2/berkeley-db-18.1.40-eybdxegfzsgulufzpjmxoq7na3xhlafw/lib;/public/soft/linux-centos7-x86_64/intel-2022.0.2/libsigsegv-2.13-ptpbxgbp62qh7safgt6u75fed3xec7ge/lib;/public/soft/linux-centos7-x86_64/gcc-10.3.0/openmpi-4.1.5-atnooy4j3scc3fxoqn5scmzvpjqmoa3p/lib;/public/soft/linux-centos7-x86_64/gcc-10.3.0/ucx-1.14.0-thg2ethlnec5vdv6cmpok3zg52lfkvoy/lib;/public/soft/linux-centos7-x86_64/gcc-10.3.0/pmix-4.2.3-ubyezqsqijflgxmbem7rnu4durhmxihh/lib;/public/soft/linux-centos7-x86_64/gcc-10.3.0/libevent-2.1.12-4pbdwhqbk3r2vgv54webjc4w4pwv6dqw/lib;/public/soft/linux-centos7-x86_64/gcc-10.3.0/hwloc-2.9.1-n35tnosj7xojsolcwnoamonp2zgk3xo2/lib;/public/soft/linux-centos7-x86_64/gcc-10.3.0/cuda-11.4.4-uizl3zvwy66u3bqllanvuxwxxwfyytwo/lib64;/public/soft/linux-centos7-x86_64/gcc-10.3.0/miniconda3-4.10.3-cywznglw5bpikxtaolfbb3vvo6uq2dtx/lib;/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_MT "")
