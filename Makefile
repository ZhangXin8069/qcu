# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

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
CMAKE_COMMAND = /public/home/zhangxin/dcu/miniconda3/envs/qcu/bin/cmake

# The command to remove a file.
RM = /public/home/zhangxin/dcu/miniconda3/envs/qcu/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /public/home/zhangxin/dcu/qcu

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /public/home/zhangxin/dcu/qcu

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/public/home/zhangxin/dcu/miniconda3/envs/qcu/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/public/home/zhangxin/dcu/miniconda3/envs/qcu/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /public/home/zhangxin/dcu/qcu/CMakeFiles /public/home/zhangxin/dcu/qcu//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /public/home/zhangxin/dcu/qcu/CMakeFiles 0
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

src/mpi_clover_bistabcg.o: src/mpi_clover_bistabcg.cu.o
.PHONY : src/mpi_clover_bistabcg.o

# target to build an object file
src/mpi_clover_bistabcg.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_clover_bistabcg.cu.o
.PHONY : src/mpi_clover_bistabcg.cu.o

src/mpi_clover_bistabcg.i: src/mpi_clover_bistabcg.cu.i
.PHONY : src/mpi_clover_bistabcg.i

# target to preprocess a source file
src/mpi_clover_bistabcg.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_clover_bistabcg.cu.i
.PHONY : src/mpi_clover_bistabcg.cu.i

src/mpi_clover_bistabcg.s: src/mpi_clover_bistabcg.cu.s
.PHONY : src/mpi_clover_bistabcg.s

# target to generate assembly for a file
src/mpi_clover_bistabcg.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_clover_bistabcg.cu.s
.PHONY : src/mpi_clover_bistabcg.cu.s

src/mpi_clover_dslash.o: src/mpi_clover_dslash.cu.o
.PHONY : src/mpi_clover_dslash.o

# target to build an object file
src/mpi_clover_dslash.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_clover_dslash.cu.o
.PHONY : src/mpi_clover_dslash.cu.o

src/mpi_clover_dslash.i: src/mpi_clover_dslash.cu.i
.PHONY : src/mpi_clover_dslash.i

# target to preprocess a source file
src/mpi_clover_dslash.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_clover_dslash.cu.i
.PHONY : src/mpi_clover_dslash.cu.i

src/mpi_clover_dslash.s: src/mpi_clover_dslash.cu.s
.PHONY : src/mpi_clover_dslash.s

# target to generate assembly for a file
src/mpi_clover_dslash.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_clover_dslash.cu.s
.PHONY : src/mpi_clover_dslash.cu.s

src/mpi_clover_multgrid.o: src/mpi_clover_multgrid.cu.o
.PHONY : src/mpi_clover_multgrid.o

# target to build an object file
src/mpi_clover_multgrid.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_clover_multgrid.cu.o
.PHONY : src/mpi_clover_multgrid.cu.o

src/mpi_clover_multgrid.i: src/mpi_clover_multgrid.cu.i
.PHONY : src/mpi_clover_multgrid.i

# target to preprocess a source file
src/mpi_clover_multgrid.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_clover_multgrid.cu.i
.PHONY : src/mpi_clover_multgrid.cu.i

src/mpi_clover_multgrid.s: src/mpi_clover_multgrid.cu.s
.PHONY : src/mpi_clover_multgrid.s

# target to generate assembly for a file
src/mpi_clover_multgrid.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_clover_multgrid.cu.s
.PHONY : src/mpi_clover_multgrid.cu.s

src/mpi_overlap_bistabcg.o: src/mpi_overlap_bistabcg.cu.o
.PHONY : src/mpi_overlap_bistabcg.o

# target to build an object file
src/mpi_overlap_bistabcg.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_overlap_bistabcg.cu.o
.PHONY : src/mpi_overlap_bistabcg.cu.o

src/mpi_overlap_bistabcg.i: src/mpi_overlap_bistabcg.cu.i
.PHONY : src/mpi_overlap_bistabcg.i

# target to preprocess a source file
src/mpi_overlap_bistabcg.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_overlap_bistabcg.cu.i
.PHONY : src/mpi_overlap_bistabcg.cu.i

src/mpi_overlap_bistabcg.s: src/mpi_overlap_bistabcg.cu.s
.PHONY : src/mpi_overlap_bistabcg.s

# target to generate assembly for a file
src/mpi_overlap_bistabcg.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_overlap_bistabcg.cu.s
.PHONY : src/mpi_overlap_bistabcg.cu.s

src/mpi_overlap_dslash.o: src/mpi_overlap_dslash.cu.o
.PHONY : src/mpi_overlap_dslash.o

# target to build an object file
src/mpi_overlap_dslash.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_overlap_dslash.cu.o
.PHONY : src/mpi_overlap_dslash.cu.o

src/mpi_overlap_dslash.i: src/mpi_overlap_dslash.cu.i
.PHONY : src/mpi_overlap_dslash.i

# target to preprocess a source file
src/mpi_overlap_dslash.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_overlap_dslash.cu.i
.PHONY : src/mpi_overlap_dslash.cu.i

src/mpi_overlap_dslash.s: src/mpi_overlap_dslash.cu.s
.PHONY : src/mpi_overlap_dslash.s

# target to generate assembly for a file
src/mpi_overlap_dslash.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_overlap_dslash.cu.s
.PHONY : src/mpi_overlap_dslash.cu.s

src/mpi_overlap_multgrid.o: src/mpi_overlap_multgrid.cu.o
.PHONY : src/mpi_overlap_multgrid.o

# target to build an object file
src/mpi_overlap_multgrid.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_overlap_multgrid.cu.o
.PHONY : src/mpi_overlap_multgrid.cu.o

src/mpi_overlap_multgrid.i: src/mpi_overlap_multgrid.cu.i
.PHONY : src/mpi_overlap_multgrid.i

# target to preprocess a source file
src/mpi_overlap_multgrid.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_overlap_multgrid.cu.i
.PHONY : src/mpi_overlap_multgrid.cu.i

src/mpi_overlap_multgrid.s: src/mpi_overlap_multgrid.cu.s
.PHONY : src/mpi_overlap_multgrid.s

# target to generate assembly for a file
src/mpi_overlap_multgrid.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_overlap_multgrid.cu.s
.PHONY : src/mpi_overlap_multgrid.cu.s

src/mpi_wilson_bistabcg.o: src/mpi_wilson_bistabcg.cu.o
.PHONY : src/mpi_wilson_bistabcg.o

# target to build an object file
src/mpi_wilson_bistabcg.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_wilson_bistabcg.cu.o
.PHONY : src/mpi_wilson_bistabcg.cu.o

src/mpi_wilson_bistabcg.i: src/mpi_wilson_bistabcg.cu.i
.PHONY : src/mpi_wilson_bistabcg.i

# target to preprocess a source file
src/mpi_wilson_bistabcg.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_wilson_bistabcg.cu.i
.PHONY : src/mpi_wilson_bistabcg.cu.i

src/mpi_wilson_bistabcg.s: src/mpi_wilson_bistabcg.cu.s
.PHONY : src/mpi_wilson_bistabcg.s

# target to generate assembly for a file
src/mpi_wilson_bistabcg.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_wilson_bistabcg.cu.s
.PHONY : src/mpi_wilson_bistabcg.cu.s

src/mpi_wilson_dslash.o: src/mpi_wilson_dslash.cu.o
.PHONY : src/mpi_wilson_dslash.o

# target to build an object file
src/mpi_wilson_dslash.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_wilson_dslash.cu.o
.PHONY : src/mpi_wilson_dslash.cu.o

src/mpi_wilson_dslash.i: src/mpi_wilson_dslash.cu.i
.PHONY : src/mpi_wilson_dslash.i

# target to preprocess a source file
src/mpi_wilson_dslash.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_wilson_dslash.cu.i
.PHONY : src/mpi_wilson_dslash.cu.i

src/mpi_wilson_dslash.s: src/mpi_wilson_dslash.cu.s
.PHONY : src/mpi_wilson_dslash.s

# target to generate assembly for a file
src/mpi_wilson_dslash.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_wilson_dslash.cu.s
.PHONY : src/mpi_wilson_dslash.cu.s

src/mpi_wilson_multgrid.o: src/mpi_wilson_multgrid.cu.o
.PHONY : src/mpi_wilson_multgrid.o

# target to build an object file
src/mpi_wilson_multgrid.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_wilson_multgrid.cu.o
.PHONY : src/mpi_wilson_multgrid.cu.o

src/mpi_wilson_multgrid.i: src/mpi_wilson_multgrid.cu.i
.PHONY : src/mpi_wilson_multgrid.i

# target to preprocess a source file
src/mpi_wilson_multgrid.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_wilson_multgrid.cu.i
.PHONY : src/mpi_wilson_multgrid.cu.i

src/mpi_wilson_multgrid.s: src/mpi_wilson_multgrid.cu.s
.PHONY : src/mpi_wilson_multgrid.s

# target to generate assembly for a file
src/mpi_wilson_multgrid.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/mpi_wilson_multgrid.cu.s
.PHONY : src/mpi_wilson_multgrid.cu.s

src/test.o: src/test.cu.o
.PHONY : src/test.o

# target to build an object file
src/test.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/test.cu.o
.PHONY : src/test.cu.o

src/test.i: src/test.cu.i
.PHONY : src/test.i

# target to preprocess a source file
src/test.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/test.cu.i
.PHONY : src/test.cu.i

src/test.s: src/test.cu.s
.PHONY : src/test.s

# target to generate assembly for a file
src/test.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/test.cu.s
.PHONY : src/test.cu.s

src/test_bistabcg.o: src/test_bistabcg.cu.o
.PHONY : src/test_bistabcg.o

# target to build an object file
src/test_bistabcg.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/test_bistabcg.cu.o
.PHONY : src/test_bistabcg.cu.o

src/test_bistabcg.i: src/test_bistabcg.cu.i
.PHONY : src/test_bistabcg.i

# target to preprocess a source file
src/test_bistabcg.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/test_bistabcg.cu.i
.PHONY : src/test_bistabcg.cu.i

src/test_bistabcg.s: src/test_bistabcg.cu.s
.PHONY : src/test_bistabcg.s

# target to generate assembly for a file
src/test_bistabcg.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/test_bistabcg.cu.s
.PHONY : src/test_bistabcg.cu.s

src/test_dslash.o: src/test_dslash.cu.o
.PHONY : src/test_dslash.o

# target to build an object file
src/test_dslash.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/test_dslash.cu.o
.PHONY : src/test_dslash.cu.o

src/test_dslash.i: src/test_dslash.cu.i
.PHONY : src/test_dslash.i

# target to preprocess a source file
src/test_dslash.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/test_dslash.cu.i
.PHONY : src/test_dslash.cu.i

src/test_dslash.s: src/test_dslash.cu.s
.PHONY : src/test_dslash.s

# target to generate assembly for a file
src/test_dslash.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/test_dslash.cu.s
.PHONY : src/test_dslash.cu.s

src/test_multgrid.o: src/test_multgrid.cu.o
.PHONY : src/test_multgrid.o

# target to build an object file
src/test_multgrid.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/test_multgrid.cu.o
.PHONY : src/test_multgrid.cu.o

src/test_multgrid.i: src/test_multgrid.cu.i
.PHONY : src/test_multgrid.i

# target to preprocess a source file
src/test_multgrid.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/test_multgrid.cu.i
.PHONY : src/test_multgrid.cu.i

src/test_multgrid.s: src/test_multgrid.cu.s
.PHONY : src/test_multgrid.s

# target to generate assembly for a file
src/test_multgrid.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/qcu.dir/build.make CMakeFiles/qcu.dir/src/test_multgrid.cu.s
.PHONY : src/test_multgrid.cu.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... qcu"
	@echo "... src/mpi_clover_bistabcg.o"
	@echo "... src/mpi_clover_bistabcg.i"
	@echo "... src/mpi_clover_bistabcg.s"
	@echo "... src/mpi_clover_dslash.o"
	@echo "... src/mpi_clover_dslash.i"
	@echo "... src/mpi_clover_dslash.s"
	@echo "... src/mpi_clover_multgrid.o"
	@echo "... src/mpi_clover_multgrid.i"
	@echo "... src/mpi_clover_multgrid.s"
	@echo "... src/mpi_overlap_bistabcg.o"
	@echo "... src/mpi_overlap_bistabcg.i"
	@echo "... src/mpi_overlap_bistabcg.s"
	@echo "... src/mpi_overlap_dslash.o"
	@echo "... src/mpi_overlap_dslash.i"
	@echo "... src/mpi_overlap_dslash.s"
	@echo "... src/mpi_overlap_multgrid.o"
	@echo "... src/mpi_overlap_multgrid.i"
	@echo "... src/mpi_overlap_multgrid.s"
	@echo "... src/mpi_wilson_bistabcg.o"
	@echo "... src/mpi_wilson_bistabcg.i"
	@echo "... src/mpi_wilson_bistabcg.s"
	@echo "... src/mpi_wilson_dslash.o"
	@echo "... src/mpi_wilson_dslash.i"
	@echo "... src/mpi_wilson_dslash.s"
	@echo "... src/mpi_wilson_multgrid.o"
	@echo "... src/mpi_wilson_multgrid.i"
	@echo "... src/mpi_wilson_multgrid.s"
	@echo "... src/test.o"
	@echo "... src/test.i"
	@echo "... src/test.s"
	@echo "... src/test_bistabcg.o"
	@echo "... src/test_bistabcg.i"
	@echo "... src/test_bistabcg.s"
	@echo "... src/test_dslash.o"
	@echo "... src/test_dslash.i"
	@echo "... src/test_dslash.s"
	@echo "... src/test_multgrid.o"
	@echo "... src/test_multgrid.i"
	@echo "... src/test_multgrid.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

