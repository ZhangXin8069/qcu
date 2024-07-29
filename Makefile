# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

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
CMAKE_SOURCE_DIR = /home/kfutfd/qcu

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kfutfd/qcu

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/kfutfd/qcu/CMakeFiles /home/kfutfd/qcu//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/kfutfd/qcu/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named qcu

# Build rule for target.
qcu: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 qcu
.PHONY : qcu

# fast build rule for target.
qcu/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/build
.PHONY : qcu/fast

src/cuda/bistabcg.o: src/cuda/bistabcg.cu.o
.PHONY : src/cuda/bistabcg.o

# target to build an object file
src/cuda/bistabcg.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/bistabcg.cu.o
.PHONY : src/cuda/bistabcg.cu.o

src/cuda/bistabcg.i: src/cuda/bistabcg.cu.i
.PHONY : src/cuda/bistabcg.i

# target to preprocess a source file
src/cuda/bistabcg.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/bistabcg.cu.i
.PHONY : src/cuda/bistabcg.cu.i

src/cuda/bistabcg.s: src/cuda/bistabcg.cu.s
.PHONY : src/cuda/bistabcg.s

# target to generate assembly for a file
src/cuda/bistabcg.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/bistabcg.cu.s
.PHONY : src/cuda/bistabcg.cu.s

src/cuda/clover_dslash.o: src/cuda/clover_dslash.cu.o
.PHONY : src/cuda/clover_dslash.o

# target to build an object file
src/cuda/clover_dslash.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/clover_dslash.cu.o
.PHONY : src/cuda/clover_dslash.cu.o

src/cuda/clover_dslash.i: src/cuda/clover_dslash.cu.i
.PHONY : src/cuda/clover_dslash.i

# target to preprocess a source file
src/cuda/clover_dslash.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/clover_dslash.cu.i
.PHONY : src/cuda/clover_dslash.cu.i

src/cuda/clover_dslash.s: src/cuda/clover_dslash.cu.s
.PHONY : src/cuda/clover_dslash.s

# target to generate assembly for a file
src/cuda/clover_dslash.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/clover_dslash.cu.s
.PHONY : src/cuda/clover_dslash.cu.s

src/cuda/lattice_cuda.o: src/cuda/lattice_cuda.cu.o
.PHONY : src/cuda/lattice_cuda.o

# target to build an object file
src/cuda/lattice_cuda.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/lattice_cuda.cu.o
.PHONY : src/cuda/lattice_cuda.cu.o

src/cuda/lattice_cuda.i: src/cuda/lattice_cuda.cu.i
.PHONY : src/cuda/lattice_cuda.i

# target to preprocess a source file
src/cuda/lattice_cuda.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/lattice_cuda.cu.i
.PHONY : src/cuda/lattice_cuda.cu.i

src/cuda/lattice_cuda.s: src/cuda/lattice_cuda.cu.s
.PHONY : src/cuda/lattice_cuda.s

# target to generate assembly for a file
src/cuda/lattice_cuda.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/lattice_cuda.cu.s
.PHONY : src/cuda/lattice_cuda.cu.s

src/cuda/lattice_mpi.o: src/cuda/lattice_mpi.cu.o
.PHONY : src/cuda/lattice_mpi.o

# target to build an object file
src/cuda/lattice_mpi.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/lattice_mpi.cu.o
.PHONY : src/cuda/lattice_mpi.cu.o

src/cuda/lattice_mpi.i: src/cuda/lattice_mpi.cu.i
.PHONY : src/cuda/lattice_mpi.i

# target to preprocess a source file
src/cuda/lattice_mpi.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/lattice_mpi.cu.i
.PHONY : src/cuda/lattice_mpi.cu.i

src/cuda/lattice_mpi.s: src/cuda/lattice_mpi.cu.s
.PHONY : src/cuda/lattice_mpi.s

# target to generate assembly for a file
src/cuda/lattice_mpi.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/lattice_mpi.cu.s
.PHONY : src/cuda/lattice_mpi.cu.s

src/cuda/mpi_clover_bistabcg.o: src/cuda/mpi_clover_bistabcg.cu.o
.PHONY : src/cuda/mpi_clover_bistabcg.o

# target to build an object file
src/cuda/mpi_clover_bistabcg.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_clover_bistabcg.cu.o
.PHONY : src/cuda/mpi_clover_bistabcg.cu.o

src/cuda/mpi_clover_bistabcg.i: src/cuda/mpi_clover_bistabcg.cu.i
.PHONY : src/cuda/mpi_clover_bistabcg.i

# target to preprocess a source file
src/cuda/mpi_clover_bistabcg.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_clover_bistabcg.cu.i
.PHONY : src/cuda/mpi_clover_bistabcg.cu.i

src/cuda/mpi_clover_bistabcg.s: src/cuda/mpi_clover_bistabcg.cu.s
.PHONY : src/cuda/mpi_clover_bistabcg.s

# target to generate assembly for a file
src/cuda/mpi_clover_bistabcg.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_clover_bistabcg.cu.s
.PHONY : src/cuda/mpi_clover_bistabcg.cu.s

src/cuda/mpi_clover_dslash.o: src/cuda/mpi_clover_dslash.cu.o
.PHONY : src/cuda/mpi_clover_dslash.o

# target to build an object file
src/cuda/mpi_clover_dslash.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_clover_dslash.cu.o
.PHONY : src/cuda/mpi_clover_dslash.cu.o

src/cuda/mpi_clover_dslash.i: src/cuda/mpi_clover_dslash.cu.i
.PHONY : src/cuda/mpi_clover_dslash.i

# target to preprocess a source file
src/cuda/mpi_clover_dslash.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_clover_dslash.cu.i
.PHONY : src/cuda/mpi_clover_dslash.cu.i

src/cuda/mpi_clover_dslash.s: src/cuda/mpi_clover_dslash.cu.s
.PHONY : src/cuda/mpi_clover_dslash.s

# target to generate assembly for a file
src/cuda/mpi_clover_dslash.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_clover_dslash.cu.s
.PHONY : src/cuda/mpi_clover_dslash.cu.s

src/cuda/mpi_clover_multgrid.o: src/cuda/mpi_clover_multgrid.cu.o
.PHONY : src/cuda/mpi_clover_multgrid.o

# target to build an object file
src/cuda/mpi_clover_multgrid.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_clover_multgrid.cu.o
.PHONY : src/cuda/mpi_clover_multgrid.cu.o

src/cuda/mpi_clover_multgrid.i: src/cuda/mpi_clover_multgrid.cu.i
.PHONY : src/cuda/mpi_clover_multgrid.i

# target to preprocess a source file
src/cuda/mpi_clover_multgrid.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_clover_multgrid.cu.i
.PHONY : src/cuda/mpi_clover_multgrid.cu.i

src/cuda/mpi_clover_multgrid.s: src/cuda/mpi_clover_multgrid.cu.s
.PHONY : src/cuda/mpi_clover_multgrid.s

# target to generate assembly for a file
src/cuda/mpi_clover_multgrid.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_clover_multgrid.cu.s
.PHONY : src/cuda/mpi_clover_multgrid.cu.s

src/cuda/mpi_overlap_bistabcg.o: src/cuda/mpi_overlap_bistabcg.cu.o
.PHONY : src/cuda/mpi_overlap_bistabcg.o

# target to build an object file
src/cuda/mpi_overlap_bistabcg.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_overlap_bistabcg.cu.o
.PHONY : src/cuda/mpi_overlap_bistabcg.cu.o

src/cuda/mpi_overlap_bistabcg.i: src/cuda/mpi_overlap_bistabcg.cu.i
.PHONY : src/cuda/mpi_overlap_bistabcg.i

# target to preprocess a source file
src/cuda/mpi_overlap_bistabcg.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_overlap_bistabcg.cu.i
.PHONY : src/cuda/mpi_overlap_bistabcg.cu.i

src/cuda/mpi_overlap_bistabcg.s: src/cuda/mpi_overlap_bistabcg.cu.s
.PHONY : src/cuda/mpi_overlap_bistabcg.s

# target to generate assembly for a file
src/cuda/mpi_overlap_bistabcg.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_overlap_bistabcg.cu.s
.PHONY : src/cuda/mpi_overlap_bistabcg.cu.s

src/cuda/mpi_overlap_dslash.o: src/cuda/mpi_overlap_dslash.cu.o
.PHONY : src/cuda/mpi_overlap_dslash.o

# target to build an object file
src/cuda/mpi_overlap_dslash.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_overlap_dslash.cu.o
.PHONY : src/cuda/mpi_overlap_dslash.cu.o

src/cuda/mpi_overlap_dslash.i: src/cuda/mpi_overlap_dslash.cu.i
.PHONY : src/cuda/mpi_overlap_dslash.i

# target to preprocess a source file
src/cuda/mpi_overlap_dslash.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_overlap_dslash.cu.i
.PHONY : src/cuda/mpi_overlap_dslash.cu.i

src/cuda/mpi_overlap_dslash.s: src/cuda/mpi_overlap_dslash.cu.s
.PHONY : src/cuda/mpi_overlap_dslash.s

# target to generate assembly for a file
src/cuda/mpi_overlap_dslash.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_overlap_dslash.cu.s
.PHONY : src/cuda/mpi_overlap_dslash.cu.s

src/cuda/mpi_overlap_multgrid.o: src/cuda/mpi_overlap_multgrid.cu.o
.PHONY : src/cuda/mpi_overlap_multgrid.o

# target to build an object file
src/cuda/mpi_overlap_multgrid.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_overlap_multgrid.cu.o
.PHONY : src/cuda/mpi_overlap_multgrid.cu.o

src/cuda/mpi_overlap_multgrid.i: src/cuda/mpi_overlap_multgrid.cu.i
.PHONY : src/cuda/mpi_overlap_multgrid.i

# target to preprocess a source file
src/cuda/mpi_overlap_multgrid.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_overlap_multgrid.cu.i
.PHONY : src/cuda/mpi_overlap_multgrid.cu.i

src/cuda/mpi_overlap_multgrid.s: src/cuda/mpi_overlap_multgrid.cu.s
.PHONY : src/cuda/mpi_overlap_multgrid.s

# target to generate assembly for a file
src/cuda/mpi_overlap_multgrid.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_overlap_multgrid.cu.s
.PHONY : src/cuda/mpi_overlap_multgrid.cu.s

src/cuda/mpi_wilson_bistabcg.o: src/cuda/mpi_wilson_bistabcg.cu.o
.PHONY : src/cuda/mpi_wilson_bistabcg.o

# target to build an object file
src/cuda/mpi_wilson_bistabcg.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_wilson_bistabcg.cu.o
.PHONY : src/cuda/mpi_wilson_bistabcg.cu.o

src/cuda/mpi_wilson_bistabcg.i: src/cuda/mpi_wilson_bistabcg.cu.i
.PHONY : src/cuda/mpi_wilson_bistabcg.i

# target to preprocess a source file
src/cuda/mpi_wilson_bistabcg.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_wilson_bistabcg.cu.i
.PHONY : src/cuda/mpi_wilson_bistabcg.cu.i

src/cuda/mpi_wilson_bistabcg.s: src/cuda/mpi_wilson_bistabcg.cu.s
.PHONY : src/cuda/mpi_wilson_bistabcg.s

# target to generate assembly for a file
src/cuda/mpi_wilson_bistabcg.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_wilson_bistabcg.cu.s
.PHONY : src/cuda/mpi_wilson_bistabcg.cu.s

src/cuda/mpi_wilson_dslash.o: src/cuda/mpi_wilson_dslash.cu.o
.PHONY : src/cuda/mpi_wilson_dslash.o

# target to build an object file
src/cuda/mpi_wilson_dslash.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_wilson_dslash.cu.o
.PHONY : src/cuda/mpi_wilson_dslash.cu.o

src/cuda/mpi_wilson_dslash.i: src/cuda/mpi_wilson_dslash.cu.i
.PHONY : src/cuda/mpi_wilson_dslash.i

# target to preprocess a source file
src/cuda/mpi_wilson_dslash.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_wilson_dslash.cu.i
.PHONY : src/cuda/mpi_wilson_dslash.cu.i

src/cuda/mpi_wilson_dslash.s: src/cuda/mpi_wilson_dslash.cu.s
.PHONY : src/cuda/mpi_wilson_dslash.s

# target to generate assembly for a file
src/cuda/mpi_wilson_dslash.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_wilson_dslash.cu.s
.PHONY : src/cuda/mpi_wilson_dslash.cu.s

src/cuda/mpi_wilson_multgrid.o: src/cuda/mpi_wilson_multgrid.cu.o
.PHONY : src/cuda/mpi_wilson_multgrid.o

# target to build an object file
src/cuda/mpi_wilson_multgrid.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_wilson_multgrid.cu.o
.PHONY : src/cuda/mpi_wilson_multgrid.cu.o

src/cuda/mpi_wilson_multgrid.i: src/cuda/mpi_wilson_multgrid.cu.i
.PHONY : src/cuda/mpi_wilson_multgrid.i

# target to preprocess a source file
src/cuda/mpi_wilson_multgrid.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_wilson_multgrid.cu.i
.PHONY : src/cuda/mpi_wilson_multgrid.cu.i

src/cuda/mpi_wilson_multgrid.s: src/cuda/mpi_wilson_multgrid.cu.s
.PHONY : src/cuda/mpi_wilson_multgrid.s

# target to generate assembly for a file
src/cuda/mpi_wilson_multgrid.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_wilson_multgrid.cu.s
.PHONY : src/cuda/mpi_wilson_multgrid.cu.s

src/cuda/multgrid.o: src/cuda/multgrid.cu.o
.PHONY : src/cuda/multgrid.o

# target to build an object file
src/cuda/multgrid.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/multgrid.cu.o
.PHONY : src/cuda/multgrid.cu.o

src/cuda/multgrid.i: src/cuda/multgrid.cu.i
.PHONY : src/cuda/multgrid.i

# target to preprocess a source file
src/cuda/multgrid.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/multgrid.cu.i
.PHONY : src/cuda/multgrid.cu.i

src/cuda/multgrid.s: src/cuda/multgrid.cu.s
.PHONY : src/cuda/multgrid.s

# target to generate assembly for a file
src/cuda/multgrid.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/multgrid.cu.s
.PHONY : src/cuda/multgrid.cu.s

src/cuda/nccl_wilson_bistabcg.o: src/cuda/nccl_wilson_bistabcg.cu.o
.PHONY : src/cuda/nccl_wilson_bistabcg.o

# target to build an object file
src/cuda/nccl_wilson_bistabcg.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/nccl_wilson_bistabcg.cu.o
.PHONY : src/cuda/nccl_wilson_bistabcg.cu.o

src/cuda/nccl_wilson_bistabcg.i: src/cuda/nccl_wilson_bistabcg.cu.i
.PHONY : src/cuda/nccl_wilson_bistabcg.i

# target to preprocess a source file
src/cuda/nccl_wilson_bistabcg.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/nccl_wilson_bistabcg.cu.i
.PHONY : src/cuda/nccl_wilson_bistabcg.cu.i

src/cuda/nccl_wilson_bistabcg.s: src/cuda/nccl_wilson_bistabcg.cu.s
.PHONY : src/cuda/nccl_wilson_bistabcg.s

# target to generate assembly for a file
src/cuda/nccl_wilson_bistabcg.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/nccl_wilson_bistabcg.cu.s
.PHONY : src/cuda/nccl_wilson_bistabcg.cu.s

src/cuda/nccl_wilson_dslash.o: src/cuda/nccl_wilson_dslash.cu.o
.PHONY : src/cuda/nccl_wilson_dslash.o

# target to build an object file
src/cuda/nccl_wilson_dslash.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/nccl_wilson_dslash.cu.o
.PHONY : src/cuda/nccl_wilson_dslash.cu.o

src/cuda/nccl_wilson_dslash.i: src/cuda/nccl_wilson_dslash.cu.i
.PHONY : src/cuda/nccl_wilson_dslash.i

# target to preprocess a source file
src/cuda/nccl_wilson_dslash.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/nccl_wilson_dslash.cu.i
.PHONY : src/cuda/nccl_wilson_dslash.cu.i

src/cuda/nccl_wilson_dslash.s: src/cuda/nccl_wilson_dslash.cu.s
.PHONY : src/cuda/nccl_wilson_dslash.s

# target to generate assembly for a file
src/cuda/nccl_wilson_dslash.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/nccl_wilson_dslash.cu.s
.PHONY : src/cuda/nccl_wilson_dslash.cu.s

src/cuda/test_clover_bistabcg.o: src/cuda/test_clover_bistabcg.cu.o
.PHONY : src/cuda/test_clover_bistabcg.o

# target to build an object file
src/cuda/test_clover_bistabcg.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_clover_bistabcg.cu.o
.PHONY : src/cuda/test_clover_bistabcg.cu.o

src/cuda/test_clover_bistabcg.i: src/cuda/test_clover_bistabcg.cu.i
.PHONY : src/cuda/test_clover_bistabcg.i

# target to preprocess a source file
src/cuda/test_clover_bistabcg.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_clover_bistabcg.cu.i
.PHONY : src/cuda/test_clover_bistabcg.cu.i

src/cuda/test_clover_bistabcg.s: src/cuda/test_clover_bistabcg.cu.s
.PHONY : src/cuda/test_clover_bistabcg.s

# target to generate assembly for a file
src/cuda/test_clover_bistabcg.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_clover_bistabcg.cu.s
.PHONY : src/cuda/test_clover_bistabcg.cu.s

src/cuda/test_clover_dslash.o: src/cuda/test_clover_dslash.cu.o
.PHONY : src/cuda/test_clover_dslash.o

# target to build an object file
src/cuda/test_clover_dslash.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_clover_dslash.cu.o
.PHONY : src/cuda/test_clover_dslash.cu.o

src/cuda/test_clover_dslash.i: src/cuda/test_clover_dslash.cu.i
.PHONY : src/cuda/test_clover_dslash.i

# target to preprocess a source file
src/cuda/test_clover_dslash.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_clover_dslash.cu.i
.PHONY : src/cuda/test_clover_dslash.cu.i

src/cuda/test_clover_dslash.s: src/cuda/test_clover_dslash.cu.s
.PHONY : src/cuda/test_clover_dslash.s

# target to generate assembly for a file
src/cuda/test_clover_dslash.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_clover_dslash.cu.s
.PHONY : src/cuda/test_clover_dslash.cu.s

src/cuda/test_clover_multgrid.o: src/cuda/test_clover_multgrid.cu.o
.PHONY : src/cuda/test_clover_multgrid.o

# target to build an object file
src/cuda/test_clover_multgrid.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_clover_multgrid.cu.o
.PHONY : src/cuda/test_clover_multgrid.cu.o

src/cuda/test_clover_multgrid.i: src/cuda/test_clover_multgrid.cu.i
.PHONY : src/cuda/test_clover_multgrid.i

# target to preprocess a source file
src/cuda/test_clover_multgrid.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_clover_multgrid.cu.i
.PHONY : src/cuda/test_clover_multgrid.cu.i

src/cuda/test_clover_multgrid.s: src/cuda/test_clover_multgrid.cu.s
.PHONY : src/cuda/test_clover_multgrid.s

# target to generate assembly for a file
src/cuda/test_clover_multgrid.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_clover_multgrid.cu.s
.PHONY : src/cuda/test_clover_multgrid.cu.s

src/cuda/test_overlap_bistabcg.o: src/cuda/test_overlap_bistabcg.cu.o
.PHONY : src/cuda/test_overlap_bistabcg.o

# target to build an object file
src/cuda/test_overlap_bistabcg.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_overlap_bistabcg.cu.o
.PHONY : src/cuda/test_overlap_bistabcg.cu.o

src/cuda/test_overlap_bistabcg.i: src/cuda/test_overlap_bistabcg.cu.i
.PHONY : src/cuda/test_overlap_bistabcg.i

# target to preprocess a source file
src/cuda/test_overlap_bistabcg.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_overlap_bistabcg.cu.i
.PHONY : src/cuda/test_overlap_bistabcg.cu.i

src/cuda/test_overlap_bistabcg.s: src/cuda/test_overlap_bistabcg.cu.s
.PHONY : src/cuda/test_overlap_bistabcg.s

# target to generate assembly for a file
src/cuda/test_overlap_bistabcg.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_overlap_bistabcg.cu.s
.PHONY : src/cuda/test_overlap_bistabcg.cu.s

src/cuda/test_overlap_dslash.o: src/cuda/test_overlap_dslash.cu.o
.PHONY : src/cuda/test_overlap_dslash.o

# target to build an object file
src/cuda/test_overlap_dslash.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_overlap_dslash.cu.o
.PHONY : src/cuda/test_overlap_dslash.cu.o

src/cuda/test_overlap_dslash.i: src/cuda/test_overlap_dslash.cu.i
.PHONY : src/cuda/test_overlap_dslash.i

# target to preprocess a source file
src/cuda/test_overlap_dslash.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_overlap_dslash.cu.i
.PHONY : src/cuda/test_overlap_dslash.cu.i

src/cuda/test_overlap_dslash.s: src/cuda/test_overlap_dslash.cu.s
.PHONY : src/cuda/test_overlap_dslash.s

# target to generate assembly for a file
src/cuda/test_overlap_dslash.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_overlap_dslash.cu.s
.PHONY : src/cuda/test_overlap_dslash.cu.s

src/cuda/test_overlap_multgrid.o: src/cuda/test_overlap_multgrid.cu.o
.PHONY : src/cuda/test_overlap_multgrid.o

# target to build an object file
src/cuda/test_overlap_multgrid.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_overlap_multgrid.cu.o
.PHONY : src/cuda/test_overlap_multgrid.cu.o

src/cuda/test_overlap_multgrid.i: src/cuda/test_overlap_multgrid.cu.i
.PHONY : src/cuda/test_overlap_multgrid.i

# target to preprocess a source file
src/cuda/test_overlap_multgrid.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_overlap_multgrid.cu.i
.PHONY : src/cuda/test_overlap_multgrid.cu.i

src/cuda/test_overlap_multgrid.s: src/cuda/test_overlap_multgrid.cu.s
.PHONY : src/cuda/test_overlap_multgrid.s

# target to generate assembly for a file
src/cuda/test_overlap_multgrid.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_overlap_multgrid.cu.s
.PHONY : src/cuda/test_overlap_multgrid.cu.s

src/cuda/test_wilson_bistabcg.o: src/cuda/test_wilson_bistabcg.cu.o
.PHONY : src/cuda/test_wilson_bistabcg.o

# target to build an object file
src/cuda/test_wilson_bistabcg.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_wilson_bistabcg.cu.o
.PHONY : src/cuda/test_wilson_bistabcg.cu.o

src/cuda/test_wilson_bistabcg.i: src/cuda/test_wilson_bistabcg.cu.i
.PHONY : src/cuda/test_wilson_bistabcg.i

# target to preprocess a source file
src/cuda/test_wilson_bistabcg.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_wilson_bistabcg.cu.i
.PHONY : src/cuda/test_wilson_bistabcg.cu.i

src/cuda/test_wilson_bistabcg.s: src/cuda/test_wilson_bistabcg.cu.s
.PHONY : src/cuda/test_wilson_bistabcg.s

# target to generate assembly for a file
src/cuda/test_wilson_bistabcg.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_wilson_bistabcg.cu.s
.PHONY : src/cuda/test_wilson_bistabcg.cu.s

src/cuda/test_wilson_dslash.o: src/cuda/test_wilson_dslash.cu.o
.PHONY : src/cuda/test_wilson_dslash.o

# target to build an object file
src/cuda/test_wilson_dslash.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_wilson_dslash.cu.o
.PHONY : src/cuda/test_wilson_dslash.cu.o

src/cuda/test_wilson_dslash.i: src/cuda/test_wilson_dslash.cu.i
.PHONY : src/cuda/test_wilson_dslash.i

# target to preprocess a source file
src/cuda/test_wilson_dslash.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_wilson_dslash.cu.i
.PHONY : src/cuda/test_wilson_dslash.cu.i

src/cuda/test_wilson_dslash.s: src/cuda/test_wilson_dslash.cu.s
.PHONY : src/cuda/test_wilson_dslash.s

# target to generate assembly for a file
src/cuda/test_wilson_dslash.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_wilson_dslash.cu.s
.PHONY : src/cuda/test_wilson_dslash.cu.s

src/cuda/test_wilson_multgrid.o: src/cuda/test_wilson_multgrid.cu.o
.PHONY : src/cuda/test_wilson_multgrid.o

# target to build an object file
src/cuda/test_wilson_multgrid.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_wilson_multgrid.cu.o
.PHONY : src/cuda/test_wilson_multgrid.cu.o

src/cuda/test_wilson_multgrid.i: src/cuda/test_wilson_multgrid.cu.i
.PHONY : src/cuda/test_wilson_multgrid.i

# target to preprocess a source file
src/cuda/test_wilson_multgrid.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_wilson_multgrid.cu.i
.PHONY : src/cuda/test_wilson_multgrid.cu.i

src/cuda/test_wilson_multgrid.s: src/cuda/test_wilson_multgrid.cu.s
.PHONY : src/cuda/test_wilson_multgrid.s

# target to generate assembly for a file
src/cuda/test_wilson_multgrid.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/test_wilson_multgrid.cu.s
.PHONY : src/cuda/test_wilson_multgrid.cu.s

src/cuda/wilson_dslash.o: src/cuda/wilson_dslash.cu.o
.PHONY : src/cuda/wilson_dslash.o

# target to build an object file
src/cuda/wilson_dslash.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/wilson_dslash.cu.o
.PHONY : src/cuda/wilson_dslash.cu.o

src/cuda/wilson_dslash.i: src/cuda/wilson_dslash.cu.i
.PHONY : src/cuda/wilson_dslash.i

# target to preprocess a source file
src/cuda/wilson_dslash.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/wilson_dslash.cu.i
.PHONY : src/cuda/wilson_dslash.cu.i

src/cuda/wilson_dslash.s: src/cuda/wilson_dslash.cu.s
.PHONY : src/cuda/wilson_dslash.s

# target to generate assembly for a file
src/cuda/wilson_dslash.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/wilson_dslash.cu.s
.PHONY : src/cuda/wilson_dslash.cu.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... qcu"
	@echo "... src/cuda/bistabcg.o"
	@echo "... src/cuda/bistabcg.i"
	@echo "... src/cuda/bistabcg.s"
	@echo "... src/cuda/clover_dslash.o"
	@echo "... src/cuda/clover_dslash.i"
	@echo "... src/cuda/clover_dslash.s"
	@echo "... src/cuda/lattice_cuda.o"
	@echo "... src/cuda/lattice_cuda.i"
	@echo "... src/cuda/lattice_cuda.s"
	@echo "... src/cuda/lattice_mpi.o"
	@echo "... src/cuda/lattice_mpi.i"
	@echo "... src/cuda/lattice_mpi.s"
	@echo "... src/cuda/mpi_clover_bistabcg.o"
	@echo "... src/cuda/mpi_clover_bistabcg.i"
	@echo "... src/cuda/mpi_clover_bistabcg.s"
	@echo "... src/cuda/mpi_clover_dslash.o"
	@echo "... src/cuda/mpi_clover_dslash.i"
	@echo "... src/cuda/mpi_clover_dslash.s"
	@echo "... src/cuda/mpi_clover_multgrid.o"
	@echo "... src/cuda/mpi_clover_multgrid.i"
	@echo "... src/cuda/mpi_clover_multgrid.s"
	@echo "... src/cuda/mpi_overlap_bistabcg.o"
	@echo "... src/cuda/mpi_overlap_bistabcg.i"
	@echo "... src/cuda/mpi_overlap_bistabcg.s"
	@echo "... src/cuda/mpi_overlap_dslash.o"
	@echo "... src/cuda/mpi_overlap_dslash.i"
	@echo "... src/cuda/mpi_overlap_dslash.s"
	@echo "... src/cuda/mpi_overlap_multgrid.o"
	@echo "... src/cuda/mpi_overlap_multgrid.i"
	@echo "... src/cuda/mpi_overlap_multgrid.s"
	@echo "... src/cuda/mpi_wilson_bistabcg.o"
	@echo "... src/cuda/mpi_wilson_bistabcg.i"
	@echo "... src/cuda/mpi_wilson_bistabcg.s"
	@echo "... src/cuda/mpi_wilson_dslash.o"
	@echo "... src/cuda/mpi_wilson_dslash.i"
	@echo "... src/cuda/mpi_wilson_dslash.s"
	@echo "... src/cuda/mpi_wilson_multgrid.o"
	@echo "... src/cuda/mpi_wilson_multgrid.i"
	@echo "... src/cuda/mpi_wilson_multgrid.s"
	@echo "... src/cuda/multgrid.o"
	@echo "... src/cuda/multgrid.i"
	@echo "... src/cuda/multgrid.s"
	@echo "... src/cuda/nccl_wilson_bistabcg.o"
	@echo "... src/cuda/nccl_wilson_bistabcg.i"
	@echo "... src/cuda/nccl_wilson_bistabcg.s"
	@echo "... src/cuda/nccl_wilson_dslash.o"
	@echo "... src/cuda/nccl_wilson_dslash.i"
	@echo "... src/cuda/nccl_wilson_dslash.s"
	@echo "... src/cuda/test_clover_bistabcg.o"
	@echo "... src/cuda/test_clover_bistabcg.i"
	@echo "... src/cuda/test_clover_bistabcg.s"
	@echo "... src/cuda/test_clover_dslash.o"
	@echo "... src/cuda/test_clover_dslash.i"
	@echo "... src/cuda/test_clover_dslash.s"
	@echo "... src/cuda/test_clover_multgrid.o"
	@echo "... src/cuda/test_clover_multgrid.i"
	@echo "... src/cuda/test_clover_multgrid.s"
	@echo "... src/cuda/test_overlap_bistabcg.o"
	@echo "... src/cuda/test_overlap_bistabcg.i"
	@echo "... src/cuda/test_overlap_bistabcg.s"
	@echo "... src/cuda/test_overlap_dslash.o"
	@echo "... src/cuda/test_overlap_dslash.i"
	@echo "... src/cuda/test_overlap_dslash.s"
	@echo "... src/cuda/test_overlap_multgrid.o"
	@echo "... src/cuda/test_overlap_multgrid.i"
	@echo "... src/cuda/test_overlap_multgrid.s"
	@echo "... src/cuda/test_wilson_bistabcg.o"
	@echo "... src/cuda/test_wilson_bistabcg.i"
	@echo "... src/cuda/test_wilson_bistabcg.s"
	@echo "... src/cuda/test_wilson_dslash.o"
	@echo "... src/cuda/test_wilson_dslash.i"
	@echo "... src/cuda/test_wilson_dslash.s"
	@echo "... src/cuda/test_wilson_multgrid.o"
	@echo "... src/cuda/test_wilson_multgrid.i"
	@echo "... src/cuda/test_wilson_multgrid.s"
	@echo "... src/cuda/wilson_dslash.o"
	@echo "... src/cuda/wilson_dslash.i"
	@echo "... src/cuda/wilson_dslash.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

