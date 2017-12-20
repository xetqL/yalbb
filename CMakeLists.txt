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

# Tests
add_executable(build_tests_space_partitioning
        Makefile
        ${TEST_SOURCE_DIRECTORY}/test_spatial_partitioning.cpp ${INCLUDE_DIRECTORY}/utils.hpp ${INCLUDE_DIRECTORY}/spatial_bisection.hpp)

set_target_properties(build_tests_space_partitioning
    PROPERTIES 
        OUTPUT_NAME "tests_space_partitioning")

add_executable(build_tests_utils
        Makefile
        ${TEST_SOURCE_DIRECTORY}/test_utils.cpp ${INCLUDE_DIRECTORY}/utils.hpp)
set_target_properties(build_tests_utils
        PROPERTIES
        OUTPUT_NAME "tests_utils")

# Executable
add_executable(build
        Makefile
        ${EXECUTABLE_SOURCE_DIRECTORY}/nbmpi.cpp ${INCLUDE_DIRECTORY}/utils.hpp ${INCLUDE_DIRECTORY}/spatial_bisection.hpp includes/geometric_load_balancer.hpp includes/partitioner.hpp)

set_target_properties(build 
    PROPERTIES 
        OUTPUT_NAME "nbmpi")

target_link_libraries(build
        PRIVATE
        ${MPI_C_LIBRARIES})

target_link_libraries(build
        PRIVATE
        /home/xetql/Dropbox/projects/cpp/nbody/liblj/liblj.a)

target_include_directories(build
        PRIVATE
        ${MPI_C_INCLUDE_PATH})

target_include_directories(build
        PRIVATE
        ../liblj)

add_custom_target(tests
        COMMAND bin/tests_space_partitioning
        COMMAND bin/tests_utils
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Run space partitioning tests in ${CMAKE_CURRENT_SOURCE_DIR}")

add_dependencies(tests build_tests_space_partitioning build_tests_utils)


