# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.11

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/xetql/cmake-3.11.0-rc2-Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /home/xetql/cmake-3.11.0-rc2-Linux-x86_64/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xetql/Dropbox/projects/cpp/nbody/nbmpi

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xetql/Dropbox/projects/cpp/nbody/nbmpi

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/home/xetql/cmake-3.11.0-rc2-Linux-x86_64/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/home/xetql/cmake-3.11.0-rc2-Linux-x86_64/bin/cmake-gui -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/xetql/Dropbox/projects/cpp/nbody/nbmpi/CMakeFiles /home/xetql/Dropbox/projects/cpp/nbody/nbmpi/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/xetql/Dropbox/projects/cpp/nbody/nbmpi/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named build_tests_lj

# Build rule for target.
build_tests_lj: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 build_tests_lj
.PHONY : build_tests_lj

# fast build rule for target.
build_tests_lj/fast:
	$(MAKE) -f CMakeFiles/build_tests_lj.dir/build.make CMakeFiles/build_tests_lj.dir/build
.PHONY : build_tests_lj/fast

#=============================================================================
# Target rules for targets named build_tests_lb

# Build rule for target.
build_tests_lb: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 build_tests_lb
.PHONY : build_tests_lb

# fast build rule for target.
build_tests_lb/fast:
	$(MAKE) -f CMakeFiles/build_tests_lb.dir/build.make CMakeFiles/build_tests_lb.dir/build
.PHONY : build_tests_lb/fast

#=============================================================================
# Target rules for targets named build_tests_utils

# Build rule for target.
build_tests_utils: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 build_tests_utils
.PHONY : build_tests_utils

# fast build rule for target.
build_tests_utils/fast:
	$(MAKE) -f CMakeFiles/build_tests_utils.dir/build.make CMakeFiles/build_tests_utils.dir/build
.PHONY : build_tests_utils/fast

#=============================================================================
# Target rules for targets named build_tests_space_partitioning

# Build rule for target.
build_tests_space_partitioning: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 build_tests_space_partitioning
.PHONY : build_tests_space_partitioning

# fast build rule for target.
build_tests_space_partitioning/fast:
	$(MAKE) -f CMakeFiles/build_tests_space_partitioning.dir/build.make CMakeFiles/build_tests_space_partitioning.dir/build
.PHONY : build_tests_space_partitioning/fast

#=============================================================================
# Target rules for targets named build

# Build rule for target.
build: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 build
.PHONY : build

# fast build rule for target.
build/fast:
	$(MAKE) -f CMakeFiles/build.dir/build.make CMakeFiles/build.dir/build
.PHONY : build/fast

#=============================================================================
# Target rules for targets named dataset

# Build rule for target.
dataset: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 dataset
.PHONY : dataset

# fast build rule for target.
dataset/fast:
	$(MAKE) -f CMakeFiles/dataset.dir/build.make CMakeFiles/dataset.dir/build
.PHONY : dataset/fast

#=============================================================================
# Target rules for targets named full_dataset

# Build rule for target.
full_dataset: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 full_dataset
.PHONY : full_dataset

# fast build rule for target.
full_dataset/fast:
	$(MAKE) -f CMakeFiles/full_dataset.dir/build.make CMakeFiles/full_dataset.dir/build
.PHONY : full_dataset/fast

#=============================================================================
# Target rules for targets named basegain

# Build rule for target.
basegain: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 basegain
.PHONY : basegain

# fast build rule for target.
basegain/fast:
	$(MAKE) -f CMakeFiles/basegain.dir/build.make CMakeFiles/basegain.dir/build
.PHONY : basegain/fast

#=============================================================================
# Target rules for targets named tests

# Build rule for target.
tests: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 tests
.PHONY : tests

# fast build rule for target.
tests/fast:
	$(MAKE) -f CMakeFiles/tests.dir/build.make CMakeFiles/tests.dir/build
.PHONY : tests/fast

src/dataset_base_gain.o: src/dataset_base_gain.cpp.o

.PHONY : src/dataset_base_gain.o

# target to build an object file
src/dataset_base_gain.cpp.o:
	$(MAKE) -f CMakeFiles/basegain.dir/build.make CMakeFiles/basegain.dir/src/dataset_base_gain.cpp.o
.PHONY : src/dataset_base_gain.cpp.o

src/dataset_base_gain.i: src/dataset_base_gain.cpp.i

.PHONY : src/dataset_base_gain.i

# target to preprocess a source file
src/dataset_base_gain.cpp.i:
	$(MAKE) -f CMakeFiles/basegain.dir/build.make CMakeFiles/basegain.dir/src/dataset_base_gain.cpp.i
.PHONY : src/dataset_base_gain.cpp.i

src/dataset_base_gain.s: src/dataset_base_gain.cpp.s

.PHONY : src/dataset_base_gain.s

# target to generate assembly for a file
src/dataset_base_gain.cpp.s:
	$(MAKE) -f CMakeFiles/basegain.dir/build.make CMakeFiles/basegain.dir/src/dataset_base_gain.cpp.s
.PHONY : src/dataset_base_gain.cpp.s

src/dataset_builder.o: src/dataset_builder.cpp.o

.PHONY : src/dataset_builder.o

# target to build an object file
src/dataset_builder.cpp.o:
	$(MAKE) -f CMakeFiles/dataset.dir/build.make CMakeFiles/dataset.dir/src/dataset_builder.cpp.o
.PHONY : src/dataset_builder.cpp.o

src/dataset_builder.i: src/dataset_builder.cpp.i

.PHONY : src/dataset_builder.i

# target to preprocess a source file
src/dataset_builder.cpp.i:
	$(MAKE) -f CMakeFiles/dataset.dir/build.make CMakeFiles/dataset.dir/src/dataset_builder.cpp.i
.PHONY : src/dataset_builder.cpp.i

src/dataset_builder.s: src/dataset_builder.cpp.s

.PHONY : src/dataset_builder.s

# target to generate assembly for a file
src/dataset_builder.cpp.s:
	$(MAKE) -f CMakeFiles/dataset.dir/build.make CMakeFiles/dataset.dir/src/dataset_builder.cpp.s
.PHONY : src/dataset_builder.cpp.s

src/full_dataset_builder.o: src/full_dataset_builder.cpp.o

.PHONY : src/full_dataset_builder.o

# target to build an object file
src/full_dataset_builder.cpp.o:
	$(MAKE) -f CMakeFiles/full_dataset.dir/build.make CMakeFiles/full_dataset.dir/src/full_dataset_builder.cpp.o
.PHONY : src/full_dataset_builder.cpp.o

src/full_dataset_builder.i: src/full_dataset_builder.cpp.i

.PHONY : src/full_dataset_builder.i

# target to preprocess a source file
src/full_dataset_builder.cpp.i:
	$(MAKE) -f CMakeFiles/full_dataset.dir/build.make CMakeFiles/full_dataset.dir/src/full_dataset_builder.cpp.i
.PHONY : src/full_dataset_builder.cpp.i

src/full_dataset_builder.s: src/full_dataset_builder.cpp.s

.PHONY : src/full_dataset_builder.s

# target to generate assembly for a file
src/full_dataset_builder.cpp.s:
	$(MAKE) -f CMakeFiles/full_dataset.dir/build.make CMakeFiles/full_dataset.dir/src/full_dataset_builder.cpp.s
.PHONY : src/full_dataset_builder.cpp.s

src/nbmpi.o: src/nbmpi.cpp.o

.PHONY : src/nbmpi.o

# target to build an object file
src/nbmpi.cpp.o:
	$(MAKE) -f CMakeFiles/build.dir/build.make CMakeFiles/build.dir/src/nbmpi.cpp.o
.PHONY : src/nbmpi.cpp.o

src/nbmpi.i: src/nbmpi.cpp.i

.PHONY : src/nbmpi.i

# target to preprocess a source file
src/nbmpi.cpp.i:
	$(MAKE) -f CMakeFiles/build.dir/build.make CMakeFiles/build.dir/src/nbmpi.cpp.i
.PHONY : src/nbmpi.cpp.i

src/nbmpi.s: src/nbmpi.cpp.s

.PHONY : src/nbmpi.s

# target to generate assembly for a file
src/nbmpi.cpp.s:
	$(MAKE) -f CMakeFiles/build.dir/build.make CMakeFiles/build.dir/src/nbmpi.cpp.s
.PHONY : src/nbmpi.cpp.s

tests/test_lj_physics.o: tests/test_lj_physics.cpp.o

.PHONY : tests/test_lj_physics.o

# target to build an object file
tests/test_lj_physics.cpp.o:
	$(MAKE) -f CMakeFiles/build_tests_lj.dir/build.make CMakeFiles/build_tests_lj.dir/tests/test_lj_physics.cpp.o
.PHONY : tests/test_lj_physics.cpp.o

tests/test_lj_physics.i: tests/test_lj_physics.cpp.i

.PHONY : tests/test_lj_physics.i

# target to preprocess a source file
tests/test_lj_physics.cpp.i:
	$(MAKE) -f CMakeFiles/build_tests_lj.dir/build.make CMakeFiles/build_tests_lj.dir/tests/test_lj_physics.cpp.i
.PHONY : tests/test_lj_physics.cpp.i

tests/test_lj_physics.s: tests/test_lj_physics.cpp.s

.PHONY : tests/test_lj_physics.s

# target to generate assembly for a file
tests/test_lj_physics.cpp.s:
	$(MAKE) -f CMakeFiles/build_tests_lj.dir/build.make CMakeFiles/build_tests_lj.dir/tests/test_lj_physics.cpp.s
.PHONY : tests/test_lj_physics.cpp.s

tests/test_load_balancer.o: tests/test_load_balancer.cpp.o

.PHONY : tests/test_load_balancer.o

# target to build an object file
tests/test_load_balancer.cpp.o:
	$(MAKE) -f CMakeFiles/build_tests_lb.dir/build.make CMakeFiles/build_tests_lb.dir/tests/test_load_balancer.cpp.o
.PHONY : tests/test_load_balancer.cpp.o

tests/test_load_balancer.i: tests/test_load_balancer.cpp.i

.PHONY : tests/test_load_balancer.i

# target to preprocess a source file
tests/test_load_balancer.cpp.i:
	$(MAKE) -f CMakeFiles/build_tests_lb.dir/build.make CMakeFiles/build_tests_lb.dir/tests/test_load_balancer.cpp.i
.PHONY : tests/test_load_balancer.cpp.i

tests/test_load_balancer.s: tests/test_load_balancer.cpp.s

.PHONY : tests/test_load_balancer.s

# target to generate assembly for a file
tests/test_load_balancer.cpp.s:
	$(MAKE) -f CMakeFiles/build_tests_lb.dir/build.make CMakeFiles/build_tests_lb.dir/tests/test_load_balancer.cpp.s
.PHONY : tests/test_load_balancer.cpp.s

tests/test_spatial_partitioning.o: tests/test_spatial_partitioning.cpp.o

.PHONY : tests/test_spatial_partitioning.o

# target to build an object file
tests/test_spatial_partitioning.cpp.o:
	$(MAKE) -f CMakeFiles/build_tests_space_partitioning.dir/build.make CMakeFiles/build_tests_space_partitioning.dir/tests/test_spatial_partitioning.cpp.o
.PHONY : tests/test_spatial_partitioning.cpp.o

tests/test_spatial_partitioning.i: tests/test_spatial_partitioning.cpp.i

.PHONY : tests/test_spatial_partitioning.i

# target to preprocess a source file
tests/test_spatial_partitioning.cpp.i:
	$(MAKE) -f CMakeFiles/build_tests_space_partitioning.dir/build.make CMakeFiles/build_tests_space_partitioning.dir/tests/test_spatial_partitioning.cpp.i
.PHONY : tests/test_spatial_partitioning.cpp.i

tests/test_spatial_partitioning.s: tests/test_spatial_partitioning.cpp.s

.PHONY : tests/test_spatial_partitioning.s

# target to generate assembly for a file
tests/test_spatial_partitioning.cpp.s:
	$(MAKE) -f CMakeFiles/build_tests_space_partitioning.dir/build.make CMakeFiles/build_tests_space_partitioning.dir/tests/test_spatial_partitioning.cpp.s
.PHONY : tests/test_spatial_partitioning.cpp.s

tests/test_utils.o: tests/test_utils.cpp.o

.PHONY : tests/test_utils.o

# target to build an object file
tests/test_utils.cpp.o:
	$(MAKE) -f CMakeFiles/build_tests_utils.dir/build.make CMakeFiles/build_tests_utils.dir/tests/test_utils.cpp.o
.PHONY : tests/test_utils.cpp.o

tests/test_utils.i: tests/test_utils.cpp.i

.PHONY : tests/test_utils.i

# target to preprocess a source file
tests/test_utils.cpp.i:
	$(MAKE) -f CMakeFiles/build_tests_utils.dir/build.make CMakeFiles/build_tests_utils.dir/tests/test_utils.cpp.i
.PHONY : tests/test_utils.cpp.i

tests/test_utils.s: tests/test_utils.cpp.s

.PHONY : tests/test_utils.s

# target to generate assembly for a file
tests/test_utils.cpp.s:
	$(MAKE) -f CMakeFiles/build_tests_utils.dir/build.make CMakeFiles/build_tests_utils.dir/tests/test_utils.cpp.s
.PHONY : tests/test_utils.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... build_tests_lj"
	@echo "... build_tests_lb"
	@echo "... build_tests_utils"
	@echo "... build_tests_space_partitioning"
	@echo "... build"
	@echo "... dataset"
	@echo "... full_dataset"
	@echo "... basegain"
	@echo "... tests"
	@echo "... edit_cache"
	@echo "... src/dataset_base_gain.o"
	@echo "... src/dataset_base_gain.i"
	@echo "... src/dataset_base_gain.s"
	@echo "... src/dataset_builder.o"
	@echo "... src/dataset_builder.i"
	@echo "... src/dataset_builder.s"
	@echo "... src/full_dataset_builder.o"
	@echo "... src/full_dataset_builder.i"
	@echo "... src/full_dataset_builder.s"
	@echo "... src/nbmpi.o"
	@echo "... src/nbmpi.i"
	@echo "... src/nbmpi.s"
	@echo "... tests/test_lj_physics.o"
	@echo "... tests/test_lj_physics.i"
	@echo "... tests/test_lj_physics.s"
	@echo "... tests/test_load_balancer.o"
	@echo "... tests/test_load_balancer.i"
	@echo "... tests/test_load_balancer.s"
	@echo "... tests/test_spatial_partitioning.o"
	@echo "... tests/test_spatial_partitioning.i"
	@echo "... tests/test_spatial_partitioning.s"
	@echo "... tests/test_utils.o"
	@echo "... tests/test_utils.i"
	@echo "... tests/test_utils.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

