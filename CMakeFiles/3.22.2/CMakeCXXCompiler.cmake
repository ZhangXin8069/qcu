set(CMAKE_CXX_COMPILER "/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/bin/g++")
set(CMAKE_CXX_COMPILER_ARG1 "")
set(CMAKE_CXX_COMPILER_ID "GNU")
set(CMAKE_CXX_COMPILER_VERSION "10.3.0")
set(CMAKE_CXX_COMPILER_VERSION_INTERNAL "")
set(CMAKE_CXX_COMPILER_WRAPPER "")
set(CMAKE_CXX_STANDARD_COMPUTED_DEFAULT "14")
set(CMAKE_CXX_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CXX_COMPILE_FEATURES "cxx_std_98;cxx_template_template_parameters;cxx_std_11;cxx_alias_templates;cxx_alignas;cxx_alignof;cxx_attributes;cxx_auto_type;cxx_constexpr;cxx_decltype;cxx_decltype_incomplete_return_types;cxx_default_function_template_args;cxx_defaulted_functions;cxx_defaulted_move_initializers;cxx_delegating_constructors;cxx_deleted_functions;cxx_enum_forward_declarations;cxx_explicit_conversions;cxx_extended_friend_declarations;cxx_extern_templates;cxx_final;cxx_func_identifier;cxx_generalized_initializers;cxx_inheriting_constructors;cxx_inline_namespaces;cxx_lambdas;cxx_local_type_template_args;cxx_long_long_type;cxx_noexcept;cxx_nonstatic_member_init;cxx_nullptr;cxx_override;cxx_range_for;cxx_raw_string_literals;cxx_reference_qualified_functions;cxx_right_angle_brackets;cxx_rvalue_references;cxx_sizeof_member;cxx_static_assert;cxx_strong_enums;cxx_thread_local;cxx_trailing_return_types;cxx_unicode_literals;cxx_uniform_initialization;cxx_unrestricted_unions;cxx_user_literals;cxx_variadic_macros;cxx_variadic_templates;cxx_std_14;cxx_aggregate_default_initializers;cxx_attribute_deprecated;cxx_binary_literals;cxx_contextual_conversions;cxx_decltype_auto;cxx_digit_separators;cxx_generic_lambdas;cxx_lambda_init_captures;cxx_relaxed_constexpr;cxx_return_type_deduction;cxx_variable_templates;cxx_std_17;cxx_std_20")
set(CMAKE_CXX98_COMPILE_FEATURES "cxx_std_98;cxx_template_template_parameters")
set(CMAKE_CXX11_COMPILE_FEATURES "cxx_std_11;cxx_alias_templates;cxx_alignas;cxx_alignof;cxx_attributes;cxx_auto_type;cxx_constexpr;cxx_decltype;cxx_decltype_incomplete_return_types;cxx_default_function_template_args;cxx_defaulted_functions;cxx_defaulted_move_initializers;cxx_delegating_constructors;cxx_deleted_functions;cxx_enum_forward_declarations;cxx_explicit_conversions;cxx_extended_friend_declarations;cxx_extern_templates;cxx_final;cxx_func_identifier;cxx_generalized_initializers;cxx_inheriting_constructors;cxx_inline_namespaces;cxx_lambdas;cxx_local_type_template_args;cxx_long_long_type;cxx_noexcept;cxx_nonstatic_member_init;cxx_nullptr;cxx_override;cxx_range_for;cxx_raw_string_literals;cxx_reference_qualified_functions;cxx_right_angle_brackets;cxx_rvalue_references;cxx_sizeof_member;cxx_static_assert;cxx_strong_enums;cxx_thread_local;cxx_trailing_return_types;cxx_unicode_literals;cxx_uniform_initialization;cxx_unrestricted_unions;cxx_user_literals;cxx_variadic_macros;cxx_variadic_templates")
set(CMAKE_CXX14_COMPILE_FEATURES "cxx_std_14;cxx_aggregate_default_initializers;cxx_attribute_deprecated;cxx_binary_literals;cxx_contextual_conversions;cxx_decltype_auto;cxx_digit_separators;cxx_generic_lambdas;cxx_lambda_init_captures;cxx_relaxed_constexpr;cxx_return_type_deduction;cxx_variable_templates")
set(CMAKE_CXX17_COMPILE_FEATURES "cxx_std_17")
set(CMAKE_CXX20_COMPILE_FEATURES "cxx_std_20")
set(CMAKE_CXX23_COMPILE_FEATURES "")

set(CMAKE_CXX_PLATFORM_ID "Linux")
set(CMAKE_CXX_SIMULATE_ID "")
set(CMAKE_CXX_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CXX_SIMULATE_VERSION "")




set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_CXX_COMPILER_AR "/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/bin/gcc-ar")
set(CMAKE_RANLIB "/usr/bin/ranlib")
set(CMAKE_CXX_COMPILER_RANLIB "/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/bin/gcc-ranlib")
set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_MT "")
set(CMAKE_COMPILER_IS_GNUCXX 1)
set(CMAKE_CXX_COMPILER_LOADED 1)
set(CMAKE_CXX_COMPILER_WORKS TRUE)
set(CMAKE_CXX_ABI_COMPILED TRUE)

set(CMAKE_CXX_COMPILER_ENV_VAR "CXX")

set(CMAKE_CXX_COMPILER_ID_RUN 1)
set(CMAKE_CXX_SOURCE_FILE_EXTENSIONS C;M;c++;cc;cpp;cxx;m;mm;mpp;CPP;ixx;cppm)
set(CMAKE_CXX_IGNORE_EXTENSIONS inl;h;hpp;HPP;H;o;O;obj;OBJ;def;DEF;rc;RC)

foreach (lang C OBJC OBJCXX)
  if (CMAKE_${lang}_COMPILER_ID_RUN)
    foreach(extension IN LISTS CMAKE_${lang}_SOURCE_FILE_EXTENSIONS)
      list(REMOVE_ITEM CMAKE_CXX_SOURCE_FILE_EXTENSIONS ${extension})
    endforeach()
  endif()
endforeach()

set(CMAKE_CXX_LINKER_PREFERENCE 30)
set(CMAKE_CXX_LINKER_PREFERENCE_PROPAGATES 1)

# Save compiler ABI information.
set(CMAKE_CXX_SIZEOF_DATA_PTR "8")
set(CMAKE_CXX_COMPILER_ABI "ELF")
set(CMAKE_CXX_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CXX_LIBRARY_ARCHITECTURE "")

if(CMAKE_CXX_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CXX_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CXX_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CXX_COMPILER_ABI}")
endif()

if(CMAKE_CXX_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX "")
if(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX)
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_CXX_CL_SHOWINCLUDES_PREFIX}")
endif()





set(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "/public/soft/linux-centos7-x86_64/intel-2022.0.2/zlib-1.2.11-b7lezuxwlgr5na2wyzyuluy5ccvdsubu/include;/public/soft/linux-centos7-x86_64/intel-2022.0.2/gdbm-1.19-d6biocotphllwmxpcv73dj6i2tfg76wg/include;/public/soft/linux-centos7-x86_64/intel-2022.0.2/readline-8.1-rdbh6eyrwvgk7puiuy73bm6wfsuhpekf/include;/public/soft/linux-centos7-x86_64/intel-2022.0.2/ncurses-6.2-6bbomi7ea6lf6fbo4htoipiu6dvaxsip/include;/public/soft/linux-centos7-x86_64/intel-2022.0.2/bzip2-1.0.8-5sh6f2asi3gazgfsghv5delerfqmkpiz/include;/public/soft/linux-centos7-x86_64/intel-2022.0.2/berkeley-db-18.1.40-eybdxegfzsgulufzpjmxoq7na3xhlafw/include;/public/soft/linux-centos7-x86_64/intel-2022.0.2/libsigsegv-2.13-ptpbxgbp62qh7safgt6u75fed3xec7ge/include;/public/soft/linux-centos7-x86_64/gcc-10.3.0/openmpi-4.1.5-atnooy4j3scc3fxoqn5scmzvpjqmoa3p/include;/public/soft/linux-centos7-x86_64/gcc-10.3.0/ucx-1.14.0-thg2ethlnec5vdv6cmpok3zg52lfkvoy/include;/public/soft/linux-centos7-x86_64/gcc-10.3.0/pmix-4.2.3-ubyezqsqijflgxmbem7rnu4durhmxihh/include;/public/soft/linux-centos7-x86_64/gcc-10.3.0/libevent-2.1.12-4pbdwhqbk3r2vgv54webjc4w4pwv6dqw/include;/public/soft/linux-centos7-x86_64/gcc-10.3.0/hwloc-2.9.1-n35tnosj7xojsolcwnoamonp2zgk3xo2/include;/public/soft/linux-centos7-x86_64/gcc-10.3.0/cuda-11.4.4-uizl3zvwy66u3bqllanvuxwxxwfyytwo/include;/public/soft/linux-centos7-x86_64/gcc-10.3.0/miniconda3-4.10.3-cywznglw5bpikxtaolfbb3vvo6uq2dtx/include;/public/sugon/software/compiler/dtk-23.04/include;/public/sugon/software/compiler/dtk-23.04/llvm/include;/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/include/c++/10.3.0;/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/include/c++/10.3.0/x86_64-pc-linux-gnu;/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/include/c++/10.3.0/backward;/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/lib/gcc/x86_64-pc-linux-gnu/10.3.0/include;/usr/local/include;/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/include;/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/lib/gcc/x86_64-pc-linux-gnu/10.3.0/include-fixed;/usr/include")
set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/lib64;/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/lib/gcc/x86_64-pc-linux-gnu/10.3.0;/lib64;/usr/lib64;/public/soft/linux-centos7-x86_64/intel-2022.0.2/perl-5.34.0-ufjmy3aryaitjhpvw5ilo2t3auuyqtbv/lib;/public/soft/linux-centos7-x86_64/intel-2022.0.2/zlib-1.2.11-b7lezuxwlgr5na2wyzyuluy5ccvdsubu/lib;/public/soft/linux-centos7-x86_64/intel-2022.0.2/gdbm-1.19-d6biocotphllwmxpcv73dj6i2tfg76wg/lib;/public/soft/linux-centos7-x86_64/intel-2022.0.2/readline-8.1-rdbh6eyrwvgk7puiuy73bm6wfsuhpekf/lib;/public/soft/linux-centos7-x86_64/intel-2022.0.2/ncurses-6.2-6bbomi7ea6lf6fbo4htoipiu6dvaxsip/lib;/public/soft/linux-centos7-x86_64/intel-2022.0.2/bzip2-1.0.8-5sh6f2asi3gazgfsghv5delerfqmkpiz/lib;/public/soft/linux-centos7-x86_64/intel-2022.0.2/berkeley-db-18.1.40-eybdxegfzsgulufzpjmxoq7na3xhlafw/lib;/public/soft/linux-centos7-x86_64/intel-2022.0.2/libsigsegv-2.13-ptpbxgbp62qh7safgt6u75fed3xec7ge/lib;/public/soft/linux-centos7-x86_64/gcc-10.3.0/openmpi-4.1.5-atnooy4j3scc3fxoqn5scmzvpjqmoa3p/lib;/public/soft/linux-centos7-x86_64/gcc-10.3.0/ucx-1.14.0-thg2ethlnec5vdv6cmpok3zg52lfkvoy/lib;/public/soft/linux-centos7-x86_64/gcc-10.3.0/pmix-4.2.3-ubyezqsqijflgxmbem7rnu4durhmxihh/lib;/public/soft/linux-centos7-x86_64/gcc-10.3.0/libevent-2.1.12-4pbdwhqbk3r2vgv54webjc4w4pwv6dqw/lib;/public/soft/linux-centos7-x86_64/gcc-10.3.0/hwloc-2.9.1-n35tnosj7xojsolcwnoamonp2zgk3xo2/lib;/public/soft/linux-centos7-x86_64/gcc-10.3.0/cuda-11.4.4-uizl3zvwy66u3bqllanvuxwxxwfyytwo/lib64;/public/soft/linux-centos7-x86_64/gcc-10.3.0/miniconda3-4.10.3-cywznglw5bpikxtaolfbb3vvo6uq2dtx/lib;/public/soft/linux-centos7-x86_64/gcc-4.8.5/gcc-10.3.0-uoicdrf766usj4ma5wxqq4zaqgatfyy3/lib")
set(CMAKE_CXX_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
