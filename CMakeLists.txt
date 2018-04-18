cmake_minimum_required(VERSION 3.5)
project(nbmpi)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 14)

find_package(MPI REQUIRED)
find_package(GSL REQUIRED)

set(MPI_COMPILER mpic++ CACHE String "MPI compiler binary (default is mpic++)")

set(CMAKE_CXX_COMPILER ${MPI_COMPILER})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_SOURCE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)
set(TEST_SOURCE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
set(INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/includes)

include_directories(/home/xetql/boost_1_66_0)
link_directories(/home/xetql/boost_1_66_0/stage/lib)

########################################################################################################################
# Tests
########################################################################################################################
## Space Partitioning
add_executable(build_tests_space_partitioning
        Makefile
        ${TEST_SOURCE_DIRECTORY}/test_spatial_partitioning.cpp
        ${INCLUDE_DIRECTORY}/utils.hpp
        ${INCLUDE_DIRECTORY}/spatial_bisection.hpp
        includes/spatial_elements.hpp)

set_target_properties(build_tests_space_partitioning
        PROPERTIES
        OUTPUT_NAME "tests_space_partitioning")

## Utils
add_executable(build_tests_utils
        Makefile
        ${TEST_SOURCE_DIRECTORY}/test_utils.cpp
        ${INCLUDE_DIRECTORY}/utils.hpp
        includes/spatial_elements.hpp)
set_target_properties(build_tests_utils
        PROPERTIES
        OUTPUT_NAME "tests_utils")

## Load Balancer
add_executable(build_tests_lb
        Makefile
        ${TEST_SOURCE_DIRECTORY}/test_load_balancer.cpp
        ${INCLUDE_DIRECTORY}/utils.hpp
        ${INCLUDE_DIRECTORY}/geometric_load_balancer.hpp
        ${INCLUDE_DIRECTORY}/spatial_bisection.hpp
        includes/spatial_elements.hpp)
set_target_properties(build_tests_lb
        PROPERTIES
        OUTPUT_NAME "tests_load_balancer")

## Lennard-Jones and Physics
add_executable(build_tests_lj
        Makefile
        ${TEST_SOURCE_DIRECTORY}/test_lj_physics.cpp
        ${INCLUDE_DIRECTORY}/utils.hpp
        ${INCLUDE_DIRECTORY}/geometric_load_balancer.hpp
        ${INCLUDE_DIRECTORY}/spatial_bisection.hpp
        includes/spatial_elements.hpp)
set_target_properties(build_tests_lj
        PROPERTIES
        OUTPUT_NAME "tests_lj_physics")

########################################################################################################################
# Executable
########################################################################################################################
## Main build

add_executable(build
        Makefile
        ${EXECUTABLE_SOURCE_DIRECTORY}/nbmpi.cpp
        ${INCLUDE_DIRECTORY}/utils.hpp
        ${INCLUDE_DIRECTORY}/spatial_bisection.hpp
        includes/geometric_load_balancer.hpp
        includes/partitioner.hpp
        includes/spatial_elements.hpp
        includes/ljpotential.hpp
        includes/physics.hpp
        includes/nbody_io.hpp
        includes/params.hpp
        includes/box_runner.hpp
        includes/report.hpp
        includes/graph.hpp
        includes/zoltan_fn.hpp
        includes/metrics.hpp)

set_target_properties(build 
        PROPERTIES
        OUTPUT_NAME "nbmpi")

set(ZOLTAN_LIBRARY_PATH zoltan CACHE String "zoltan library path (zoltan.a)")
set(ARMADILLO_LIBRARY_PATH "" CACHE String "armadillo library path (libarmadillo.so)")

target_link_libraries(build
        PRIVATE
        ${MPI_C_LIBRARIES}
        ${ZOLTAN_LIBRARY_PATH}
        ${ARMADILLO_LIBRARY_PATH}
        ${GSL_LIBRARIES})

set(MLPACK_INCLUDE_DIRECTORY "" CACHE String "MLpack include directory")
set(LIBRL_INCLUDE_DIRECTORY  "" CACHE String "librl include directory")
set(ARMADILLO_INCLUDE_DIRECTORY "" CACHE String "armadillo include directory")
set(ZOLTAN_INCLUDE_DIRECTORY "" CACHE String "zoltan include directory")

target_include_directories(build
        PRIVATE
        ${MPI_C_INCLUDE_PATH}
        ${MLPACK_INCLUDE_DIRECTORY}
        ${LIBRL_INCLUDE_DIRECTORY}
        ${ARMADILLO_INCLUDE_DIRECTORY}
        ${ZOLTAN_INCLUDE_DIRECTORY}
)

add_executable(dataset
        Makefile
        ${INCLUDE_DIRECTORY}/utils.hpp
        ${INCLUDE_DIRECTORY}/spatial_bisection.hpp
        includes/geometric_load_balancer.hpp
        includes/partitioner.hpp
        includes/spatial_elements.hpp
        includes/ljpotential.hpp
        includes/physics.hpp
        includes/nbody_io.hpp
        includes/params.hpp
        includes/box_runner.hpp
        includes/report.hpp
        includes/graph.hpp
        includes/zoltan_fn.hpp
        includes/metrics.hpp
        src/dataset_builder.cpp)
target_include_directories(dataset
        PRIVATE
        ${MPI_C_INCLUDE_PATH}
        ${MLPACK_INCLUDE_DIRECTORY}
        ${LIBRL_INCLUDE_DIRECTORY}
        ${ARMADILLO_INCLUDE_DIRECTORY}
        ${ZOLTAN_INCLUDE_DIRECTORY})
target_link_libraries(dataset
        PRIVATE
        ${MPI_C_LIBRARIES}
        ${ZOLTAN_LIBRARY_PATH}
        ${ARMADILLO_LIBRARY_PATH}
        ${GSL_LIBRARIES})

add_custom_target(tests
        COMMAND bin/tests_utils
        COMMAND bin/tests_space_partitioning
        COMMAND mpirun -np 2 bin/tests_load_balancer
        COMMAND bin/tests_lj_physics
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "========================"
        COMMAND ${CMAKE_COMMAND} -E echo "  Test suit completed!"
        COMMAND ${CMAKE_COMMAND} -E echo "========================"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running custom test suit for utils, space_partitioning, and load_balancer.")

add_dependencies(tests build_tests_space_partitioning build_tests_utils build_tests_lb build_tests_lj)

########################################################################################################################

