# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kfutfd/qcu/refer/qcu

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kfutfd/qcu/refer/qcu/build

# Include any dependencies generated for this target.
include CMakeFiles/test_linear_algebra.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_linear_algebra.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_linear_algebra.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_linear_algebra.dir/flags.make

CMakeFiles/test_linear_algebra.dir/src/tests/linear_algebra/test_kernels.cu.o: CMakeFiles/test_linear_algebra.dir/flags.make
CMakeFiles/test_linear_algebra.dir/src/tests/linear_algebra/test_kernels.cu.o: ../src/tests/linear_algebra/test_kernels.cu
CMakeFiles/test_linear_algebra.dir/src/tests/linear_algebra/test_kernels.cu.o: CMakeFiles/test_linear_algebra.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kfutfd/qcu/refer/qcu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/test_linear_algebra.dir/src/tests/linear_algebra/test_kernels.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/test_linear_algebra.dir/src/tests/linear_algebra/test_kernels.cu.o -MF CMakeFiles/test_linear_algebra.dir/src/tests/linear_algebra/test_kernels.cu.o.d -x cu -c /home/kfutfd/qcu/refer/qcu/src/tests/linear_algebra/test_kernels.cu -o CMakeFiles/test_linear_algebra.dir/src/tests/linear_algebra/test_kernels.cu.o

CMakeFiles/test_linear_algebra.dir/src/tests/linear_algebra/test_kernels.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/test_linear_algebra.dir/src/tests/linear_algebra/test_kernels.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/test_linear_algebra.dir/src/tests/linear_algebra/test_kernels.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/test_linear_algebra.dir/src/tests/linear_algebra/test_kernels.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target test_linear_algebra
test_linear_algebra_OBJECTS = \
"CMakeFiles/test_linear_algebra.dir/src/tests/linear_algebra/test_kernels.cu.o"

# External object files for target test_linear_algebra
test_linear_algebra_EXTERNAL_OBJECTS =

tests/test_linear_algebra: CMakeFiles/test_linear_algebra.dir/src/tests/linear_algebra/test_kernels.cu.o
tests/test_linear_algebra: CMakeFiles/test_linear_algebra.dir/build.make
tests/test_linear_algebra: libqcu.so
tests/test_linear_algebra: /home/kfutfd/external-libraries/openmpi-4.1.5/lib/libmpi.so
tests/test_linear_algebra: CMakeFiles/test_linear_algebra.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kfutfd/qcu/refer/qcu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable tests/test_linear_algebra"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_linear_algebra.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_linear_algebra.dir/build: tests/test_linear_algebra
.PHONY : CMakeFiles/test_linear_algebra.dir/build

CMakeFiles/test_linear_algebra.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_linear_algebra.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_linear_algebra.dir/clean

CMakeFiles/test_linear_algebra.dir/depend:
	cd /home/kfutfd/qcu/refer/qcu/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kfutfd/qcu/refer/qcu /home/kfutfd/qcu/refer/qcu /home/kfutfd/qcu/refer/qcu/build /home/kfutfd/qcu/refer/qcu/build /home/kfutfd/qcu/refer/qcu/build/CMakeFiles/test_linear_algebra.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_linear_algebra.dir/depend
