# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/aistudio/qcu

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/aistudio/qcu

# Include any dependencies generated for this target.
include CMakeFiles/qcu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/qcu.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/qcu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/qcu.dir/flags.make

CMakeFiles/qcu.dir/src/cuda/qcu_cuda.cu.o: CMakeFiles/qcu.dir/flags.make
CMakeFiles/qcu.dir/src/cuda/qcu_cuda.cu.o: src/cuda/qcu_cuda.cu
CMakeFiles/qcu.dir/src/cuda/qcu_cuda.cu.o: CMakeFiles/qcu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aistudio/qcu/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/qcu.dir/src/cuda/qcu_cuda.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/qcu.dir/src/cuda/qcu_cuda.cu.o -MF CMakeFiles/qcu.dir/src/cuda/qcu_cuda.cu.o.d -x cu -c /home/aistudio/qcu/src/cuda/qcu_cuda.cu -o CMakeFiles/qcu.dir/src/cuda/qcu_cuda.cu.o

CMakeFiles/qcu.dir/src/cuda/qcu_cuda.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/qcu.dir/src/cuda/qcu_cuda.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/qcu.dir/src/cuda/qcu_cuda.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/qcu.dir/src/cuda/qcu_cuda.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/qcu.dir/src/cuda/clover_dslash.cu.o: CMakeFiles/qcu.dir/flags.make
CMakeFiles/qcu.dir/src/cuda/clover_dslash.cu.o: src/cuda/clover_dslash.cu
CMakeFiles/qcu.dir/src/cuda/clover_dslash.cu.o: CMakeFiles/qcu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aistudio/qcu/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/qcu.dir/src/cuda/clover_dslash.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/qcu.dir/src/cuda/clover_dslash.cu.o -MF CMakeFiles/qcu.dir/src/cuda/clover_dslash.cu.o.d -x cu -c /home/aistudio/qcu/src/cuda/clover_dslash.cu -o CMakeFiles/qcu.dir/src/cuda/clover_dslash.cu.o

CMakeFiles/qcu.dir/src/cuda/clover_dslash.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/qcu.dir/src/cuda/clover_dslash.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/qcu.dir/src/cuda/clover_dslash.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/qcu.dir/src/cuda/clover_dslash.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/qcu.dir/src/cuda/wilson_dslash.cu.o: CMakeFiles/qcu.dir/flags.make
CMakeFiles/qcu.dir/src/cuda/wilson_dslash.cu.o: src/cuda/wilson_dslash.cu
CMakeFiles/qcu.dir/src/cuda/wilson_dslash.cu.o: CMakeFiles/qcu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aistudio/qcu/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/qcu.dir/src/cuda/wilson_dslash.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/qcu.dir/src/cuda/wilson_dslash.cu.o -MF CMakeFiles/qcu.dir/src/cuda/wilson_dslash.cu.o.d -x cu -c /home/aistudio/qcu/src/cuda/wilson_dslash.cu -o CMakeFiles/qcu.dir/src/cuda/wilson_dslash.cu.o

CMakeFiles/qcu.dir/src/cuda/wilson_dslash.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/qcu.dir/src/cuda/wilson_dslash.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/qcu.dir/src/cuda/wilson_dslash.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/qcu.dir/src/cuda/wilson_dslash.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target qcu
qcu_OBJECTS = \
"CMakeFiles/qcu.dir/src/cuda/qcu_cuda.cu.o" \
"CMakeFiles/qcu.dir/src/cuda/clover_dslash.cu.o" \
"CMakeFiles/qcu.dir/src/cuda/wilson_dslash.cu.o"

# External object files for target qcu
qcu_EXTERNAL_OBJECTS =

lib/libqcu.so: CMakeFiles/qcu.dir/src/cuda/qcu_cuda.cu.o
lib/libqcu.so: CMakeFiles/qcu.dir/src/cuda/clover_dslash.cu.o
lib/libqcu.so: CMakeFiles/qcu.dir/src/cuda/wilson_dslash.cu.o
lib/libqcu.so: CMakeFiles/qcu.dir/build.make
lib/libqcu.so: CMakeFiles/qcu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/aistudio/qcu/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CUDA shared library lib/libqcu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/qcu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/qcu.dir/build: lib/libqcu.so
.PHONY : CMakeFiles/qcu.dir/build

CMakeFiles/qcu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/qcu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/qcu.dir/clean

CMakeFiles/qcu.dir/depend:
	cd /home/aistudio/qcu && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/aistudio/qcu /home/aistudio/qcu /home/aistudio/qcu /home/aistudio/qcu /home/aistudio/qcu/CMakeFiles/qcu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/qcu.dir/depend

