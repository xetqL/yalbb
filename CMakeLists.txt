cmake_minimum_required(VERSION 3.9)
project(nbmpi)

set(CMAKE_CXX_STANDARD 17)
find_package(MPI REQUIRED)

set(CMAKE_CXX_COMPILER /usr/bin/mpic++)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_SOURCE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)
set(TEST_SOURCE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
set(INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/includes)

########################################################################################################################
# Tests

## Space Partitioning
add_executable(build_tests_space_partitioning
        Makefile
        ${TEST_SOURCE_DIRECTORY}/test_spatial_partitioning.cpp ${INCLUDE_DIRECTORY}/utils.hpp ${INCLUDE_DIRECTORY}/spatial_bisection.hpp includes/spatial_elements.hpp)

set_target_properties(build_tests_space_partitioning
        PROPERTIES
        OUTPUT_NAME "tests_space_partitioning")

## Utils
add_executable(build_tests_utils
        Makefile
        ${TEST_SOURCE_DIRECTORY}/test_utils.cpp ${INCLUDE_DIRECTORY}/utils.hpp includes/spatial_elements.hpp)
set_target_properties(build_tests_utils
        PROPERTIES
        OUTPUT_NAME "tests_utils")

## Load Balancer
add_executable(build_tests_lb
        Makefile
        ${TEST_SOURCE_DIRECTORY}/test_load_balancer.cpp ${INCLUDE_DIRECTORY}/utils.hpp ${INCLUDE_DIRECTORY}/geometric_load_balancer.hpp ${INCLUDE_DIRECTORY}/spatial_bisection.hpp includes/spatial_elements.hpp)
set_target_properties(build_tests_lb
        PROPERTIES
        OUTPUT_NAME "tests_load_balancer")

########################################################################################################################
# Executable

## Main build
add_executable(build
        Makefile
        ${EXECUTABLE_SOURCE_DIRECTORY}/nbmpi.cpp ${INCLUDE_DIRECTORY}/utils.hpp ${INCLUDE_DIRECTORY}/spatial_bisection.hpp includes/geometric_load_balancer.hpp includes/partitioner.hpp includes/spatial_elements.hpp)

set_target_properties(build 
        PROPERTIES
        OUTPUT_NAME "nbmpi")

target_link_libraries(build
        PRIVATE
        ${MPI_C_LIBRARIES})

target_link_libraries(build
        PRIVATE
        /usr/lib/liblj/liblj.a)

target_include_directories(build
        PRIVATE
        ${MPI_C_INCLUDE_PATH})

add_custom_target(tests
        COMMAND bin/tests_utils
        COMMAND bin/tests_space_partitioning
        COMMAND mpirun -np 2 bin/tests_load_balancer
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "========================"
        COMMAND ${CMAKE_COMMAND} -E echo "  Test suit completed!"
        COMMAND ${CMAKE_COMMAND} -E echo "========================"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running custom test suit for utils, space_partitioning, and load_balancer!")

add_dependencies(tests build_tests_space_partitioning build_tests_utils build_tests_lb)


