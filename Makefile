# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/aistudio/qcu

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/aistudio/qcu

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/local/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/local/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/aistudio/qcu/CMakeFiles /home/aistudio/qcu//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/aistudio/qcu/CMakeFiles 0
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

src/cuda/mpi_wilson_cg.o: src/cuda/mpi_wilson_cg.cu.o
.PHONY : src/cuda/mpi_wilson_cg.o

# target to build an object file
src/cuda/mpi_wilson_cg.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_wilson_cg.cu.o
.PHONY : src/cuda/mpi_wilson_cg.cu.o

src/cuda/mpi_wilson_cg.i: src/cuda/mpi_wilson_cg.cu.i
.PHONY : src/cuda/mpi_wilson_cg.i

# target to preprocess a source file
src/cuda/mpi_wilson_cg.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_wilson_cg.cu.i
.PHONY : src/cuda/mpi_wilson_cg.cu.i

src/cuda/mpi_wilson_cg.s: src/cuda/mpi_wilson_cg.cu.s
.PHONY : src/cuda/mpi_wilson_cg.s

# target to generate assembly for a file
src/cuda/mpi_wilson_cg.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/mpi_wilson_cg.cu.s
.PHONY : src/cuda/mpi_wilson_cg.cu.s

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

src/cuda/qcu_cuda.o: src/cuda/qcu_cuda.cu.o
.PHONY : src/cuda/qcu_cuda.o

# target to build an object file
src/cuda/qcu_cuda.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/qcu_cuda.cu.o
.PHONY : src/cuda/qcu_cuda.cu.o

src/cuda/qcu_cuda.i: src/cuda/qcu_cuda.cu.i
.PHONY : src/cuda/qcu_cuda.i

# target to preprocess a source file
src/cuda/qcu_cuda.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/qcu_cuda.cu.i
.PHONY : src/cuda/qcu_cuda.cu.i

src/cuda/qcu_cuda.s: src/cuda/qcu_cuda.cu.s
.PHONY : src/cuda/qcu_cuda.s

# target to generate assembly for a file
src/cuda/qcu_cuda.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/cuda/qcu_cuda.cu.s
.PHONY : src/cuda/qcu_cuda.cu.s

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
	@echo "... src/cuda/clover_dslash.o"
	@echo "... src/cuda/clover_dslash.i"
	@echo "... src/cuda/clover_dslash.s"
	@echo "... src/cuda/mpi_wilson_cg.o"
	@echo "... src/cuda/mpi_wilson_cg.i"
	@echo "... src/cuda/mpi_wilson_cg.s"
	@echo "... src/cuda/mpi_wilson_dslash.o"
	@echo "... src/cuda/mpi_wilson_dslash.i"
	@echo "... src/cuda/mpi_wilson_dslash.s"
	@echo "... src/cuda/qcu_cuda.o"
	@echo "... src/cuda/qcu_cuda.i"
	@echo "... src/cuda/qcu_cuda.s"
	@echo "... src/cuda/test_wilson_dslash.o"
	@echo "... src/cuda/test_wilson_dslash.i"
	@echo "... src/cuda/test_wilson_dslash.s"
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

