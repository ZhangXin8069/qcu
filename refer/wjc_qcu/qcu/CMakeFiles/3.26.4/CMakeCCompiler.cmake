set(CMAKE_C_COMPILER "/public/sugon/software/compiler/dtk-23.04/bin/hipcc")
set(CMAKE_C_COMPILER_ARG1 "")
set(CMAKE_C_COMPILER_ID "Clang")
set(CMAKE_C_COMPILER_VERSION "14.0.0")
set(CMAKE_C_COMPILER_VERSION_INTERNAL "")
set(CMAKE_C_COMPILER_WRAPPER "")
set(CMAKE_C_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_C_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_C_COMPILE_FEATURES "c_std_90;c_function_prototypes;c_std_99;c_restrict;c_variadic_macros;c_std_11;c_static_assert;c_std_17;c_std_23")
set(CMAKE_C90_COMPILE_FEATURES "c_std_90;c_function_prototypes")
set(CMAKE_C99_COMPILE_FEATURES "c_std_99;c_restrict;c_variadic_macros")
set(CMAKE_C11_COMPILE_FEATURES "c_std_11;c_static_assert")
set(CMAKE_C17_COMPILE_FEATURES "c_std_17")
set(CMAKE_C23_COMPILE_FEATURES "c_std_23")

set(CMAKE_C_PLATFORM_ID "Linux")
set(CMAKE_C_SIMULATE_ID "")
set(CMAKE_C_COMPILER_FRONTEND_VARIANT "GNU")
set(CMAKE_C_SIMULATE_VERSION "")




set(CMAKE_AR "/public/sugon/software/compiler/dtk-23.04/llvm/bin/llvm-ar")
set(CMAKE_C_COMPILER_AR "/public/sugon/software/compiler/dtk-23.04/llvm/bin/llvm-ar")
set(CMAKE_RANLIB "/public/sugon/software/compiler/dtk-23.04/llvm/bin/llvm-ranlib")
set(CMAKE_C_COMPILER_RANLIB "/public/sugon/software/compiler/dtk-23.04/llvm/bin/llvm-ranlib")
set(CMAKE_LINKER "/public/sugon/software/compiler/dtk-23.04/llvm/bin/ld.lld")
set(CMAKE_MT "")
set(CMAKE_COMPILER_IS_GNUCC )
set(CMAKE_C_COMPILER_LOADED 1)
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_C_ABI_COMPILED TRUE)

set(CMAKE_C_COMPILER_ENV_VAR "CC")

set(CMAKE_C_COMPILER_ID_RUN 1)
set(CMAKE_C_SOURCE_FILE_EXTENSIONS c;m)
set(CMAKE_C_IGNORE_EXTENSIONS h;H;o;O;obj;OBJ;def;DEF;rc;RC)
set(CMAKE_C_LINKER_PREFERENCE 10)

# Save compiler ABI information.
set(CMAKE_C_SIZEOF_DATA_PTR "8")
set(CMAKE_C_COMPILER_ABI "ELF")
set(CMAKE_C_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_C_LIBRARY_ARCHITECTURE "")

if(CMAKE_C_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_C_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_C_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_C_COMPILER_ABI}")
endif()

if(CMAKE_C_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "")
if(CMAKE_C_CL_SHOWINCLUDES_PREFIX)
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_C_CL_SHOWINCLUDES_PREFIX}")
endif()





set(CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES "/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/ompi/include;/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/ucx-without-dtk/include;/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/sharp/include;/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/hcoll/include;/public/sugon/software/compiler/dtk-23.04/llvm/lib/clang/14.0.0;/public/sugon/software/compiler/dtk-23.04/hsa/include;/public/sugon/software/compiler/dtk-23.04/hip/include;/public/sugon/software/compiler/dtk-23.04/include;/public/sugon/software/compiler/dtk-23.04/llvm/include;/public/sugon/software/compiler/gcc/7.3.1/include/c++/7;/public/sugon/software/compiler/gcc/7.3.1/include/c++/7/x86_64-pc-linux-gnu;/public/sugon/software/compiler/dtk-23.04/miopen/include;/opt/rh/devtoolset-7/root/usr/include/c++/7;/opt/rh/devtoolset-7/root/usr/include/c++/7/x86_64-redhat-linux;/opt/gridview/pmix/include;/opt/gridview/slurm/include;/opt/gridview/munge/include;/public/sugon/software/compiler/dtk-23.04/llvm/lib/clang/14.0.0/include;/usr/local/include;/usr/include")
set(CMAKE_C_IMPLICIT_LINK_LIBRARIES "gcc_s;gcc;pthread;m;rt;amdhip64;clang_rt.builtins-x86_64;stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_C_IMPLICIT_LINK_DIRECTORIES "/public/sugon/software/compiler/dtk-23.04/hip/lib;/public/sugon/software/compiler/dtk-23.04/lib;/public/sugon/software/compiler/dtk-23.04/llvm/lib/clang/14.0.0/lib/linux;/opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7;/opt/rh/devtoolset-7/root/usr/lib64;/lib64;/usr/lib64;/public/sugon/software/compiler/dtk-23.04/llvm/lib;/lib;/usr/lib;/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/ompi/lib;/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/sharp/lib;/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/hcoll/lib;/public/sugon/software/mpi/hpcx/hpcx-v2.4.1.0-gcc/ucx-without-dtk/lib;/public/sugon/software/compiler/gcc/7.3.1/lib64;/public/sugon/software/compiler/gcc/7.3.1/lib;/public/sugon/software/compiler/gcc/7.3.1/external_libs/lib")
set(CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
