# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xetql/ljmpi

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xetql/ljmpi

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/ccmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/xetql/ljmpi/CMakeFiles /home/xetql/ljmpi/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/xetql/ljmpi/CMakeFiles 0
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
# Target rules for targets named zoltan_migration

# Build rule for target.
zoltan_migration: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 zoltan_migration
.PHONY : zoltan_migration

# fast build rule for target.
zoltan_migration/fast:
	$(MAKE) -f CMakeFiles/zoltan_migration.dir/build.make CMakeFiles/zoltan_migration.dir/build
.PHONY : zoltan_migration/fast

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
# Target rules for targets named tests

# Build rule for target.
tests: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 tests
.PHONY : tests

# fast build rule for target.
tests/fast:
	$(MAKE) -f CMakeFiles/tests.dir/build.make CMakeFiles/tests.dir/build
.PHONY : tests/fast

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
# Target rules for targets named doall

# Build rule for target.
doall: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 doall
.PHONY : doall

# fast build rule for target.
doall/fast:
	$(MAKE) -f CMakeFiles/doall.dir/build.make CMakeFiles/doall.dir/build
.PHONY : doall/fast

#=============================================================================
# Target rules for targets named astar

# Build rule for target.
astar: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 astar
.PHONY : astar

# fast build rule for target.
astar/fast:
	$(MAKE) -f CMakeFiles/astar.dir/build.make CMakeFiles/astar.dir/build
.PHONY : astar/fast

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
# Target rules for targets named generate

# Build rule for target.
generate: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 generate
.PHONY : generate

# fast build rule for target.
generate/fast:
	$(MAKE) -f CMakeFiles/generate.dir/build.make CMakeFiles/generate.dir/build
.PHONY : generate/fast

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
# Target rules for targets named dataset

# Build rule for target.
dataset: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 dataset
.PHONY : dataset

# fast build rule for target.
dataset/fast:
	$(MAKE) -f CMakeFiles/dataset.dir/build.make CMakeFiles/dataset.dir/build
.PHONY : dataset/fast

src/astar_search.o: src/astar_search.cpp.o

.PHONY : src/astar_search.o

# target to build an object file
src/astar_search.cpp.o:
	$(MAKE) -f CMakeFiles/astar.dir/build.make CMakeFiles/astar.dir/src/astar_search.cpp.o
.PHONY : src/astar_search.cpp.o

src/astar_search.i: src/astar_search.cpp.i

.PHONY : src/astar_search.i

# target to preprocess a source file
src/astar_search.cpp.i:
	$(MAKE) -f CMakeFiles/astar.dir/build.make CMakeFiles/astar.dir/src/astar_search.cpp.i
.PHONY : src/astar_search.cpp.i

src/astar_search.s: src/astar_search.cpp.s

.PHONY : src/astar_search.s

# target to generate assembly for a file
src/astar_search.cpp.s:
	$(MAKE) -f CMakeFiles/astar.dir/build.make CMakeFiles/astar.dir/src/astar_search.cpp.s
.PHONY : src/astar_search.cpp.s

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

src/do_all.o: src/do_all.cpp.o

.PHONY : src/do_all.o

# target to build an object file
src/do_all.cpp.o:
	$(MAKE) -f CMakeFiles/doall.dir/build.make CMakeFiles/doall.dir/src/do_all.cpp.o
.PHONY : src/do_all.cpp.o

src/do_all.i: src/do_all.cpp.i

.PHONY : src/do_all.i

# target to preprocess a source file
src/do_all.cpp.i:
	$(MAKE) -f CMakeFiles/doall.dir/build.make CMakeFiles/doall.dir/src/do_all.cpp.i
.PHONY : src/do_all.cpp.i

src/do_all.s: src/do_all.cpp.s

.PHONY : src/do_all.s

# target to generate assembly for a file
src/do_all.cpp.s:
	$(MAKE) -f CMakeFiles/doall.dir/build.make CMakeFiles/doall.dir/src/do_all.cpp.s
.PHONY : src/do_all.cpp.s

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

src/generate.o: src/generate.cpp.o

.PHONY : src/generate.o

# target to build an object file
src/generate.cpp.o:
	$(MAKE) -f CMakeFiles/generate.dir/build.make CMakeFiles/generate.dir/src/generate.cpp.o
.PHONY : src/generate.cpp.o

src/generate.i: src/generate.cpp.i

.PHONY : src/generate.i

# target to preprocess a source file
src/generate.cpp.i:
	$(MAKE) -f CMakeFiles/generate.dir/build.make CMakeFiles/generate.dir/src/generate.cpp.i
.PHONY : src/generate.cpp.i

src/generate.s: src/generate.cpp.s

.PHONY : src/generate.s

# target to generate assembly for a file
src/generate.cpp.s:
	$(MAKE) -f CMakeFiles/generate.dir/build.make CMakeFiles/generate.dir/src/generate.cpp.s
.PHONY : src/generate.cpp.s

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

src/zoltan_migration.o: src/zoltan_migration.cpp.o

.PHONY : src/zoltan_migration.o

# target to build an object file
src/zoltan_migration.cpp.o:
	$(MAKE) -f CMakeFiles/zoltan_migration.dir/build.make CMakeFiles/zoltan_migration.dir/src/zoltan_migration.cpp.o
	$(MAKE) -f CMakeFiles/astar.dir/build.make CMakeFiles/astar.dir/src/zoltan_migration.cpp.o
.PHONY : src/zoltan_migration.cpp.o

src/zoltan_migration.i: src/zoltan_migration.cpp.i

.PHONY : src/zoltan_migration.i

# target to preprocess a source file
src/zoltan_migration.cpp.i:
	$(MAKE) -f CMakeFiles/zoltan_migration.dir/build.make CMakeFiles/zoltan_migration.dir/src/zoltan_migration.cpp.i
	$(MAKE) -f CMakeFiles/astar.dir/build.make CMakeFiles/astar.dir/src/zoltan_migration.cpp.i
.PHONY : src/zoltan_migration.cpp.i

src/zoltan_migration.s: src/zoltan_migration.cpp.s

.PHONY : src/zoltan_migration.s

# target to generate assembly for a file
src/zoltan_migration.cpp.s:
	$(MAKE) -f CMakeFiles/zoltan_migration.dir/build.make CMakeFiles/zoltan_migration.dir/src/zoltan_migration.cpp.s
	$(MAKE) -f CMakeFiles/astar.dir/build.make CMakeFiles/astar.dir/src/zoltan_migration.cpp.s
.PHONY : src/zoltan_migration.cpp.s

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
	@echo "... edit_cache"
	@echo "... zoltan_migration"
	@echo "... basegain"
	@echo "... full_dataset"
	@echo "... tests"
	@echo "... build_tests_utils"
	@echo "... doall"
	@echo "... astar"
	@echo "... build_tests_space_partitioning"
	@echo "... build"
	@echo "... build_tests_lb"
	@echo "... generate"
	@echo "... build_tests_lj"
	@echo "... dataset"
	@echo "... src/astar_search.o"
	@echo "... src/astar_search.i"
	@echo "... src/astar_search.s"
	@echo "... src/dataset_base_gain.o"
	@echo "... src/dataset_base_gain.i"
	@echo "... src/dataset_base_gain.s"
	@echo "... src/dataset_builder.o"
	@echo "... src/dataset_builder.i"
	@echo "... src/dataset_builder.s"
	@echo "... src/do_all.o"
	@echo "... src/do_all.i"
	@echo "... src/do_all.s"
	@echo "... src/full_dataset_builder.o"
	@echo "... src/full_dataset_builder.i"
	@echo "... src/full_dataset_builder.s"
	@echo "... src/generate.o"
	@echo "... src/generate.i"
	@echo "... src/generate.s"
	@echo "... src/nbmpi.o"
	@echo "... src/nbmpi.i"
	@echo "... src/nbmpi.s"
	@echo "... src/zoltan_migration.o"
	@echo "... src/zoltan_migration.i"
	@echo "... src/zoltan_migration.s"
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

