cmake_minimum_required(VERSION 3.5)
project(nbmpi)


if (CMAKE_BUILD_TYPE MATCHES "Debug")
    MESSAGE("Compile with debug flag")
    ADD_DEFINITIONS(-DDEBUG)
else ()
    set(CMAKE_BUILD_TYPE Release)
endif()

if (ASTAR_IMPL STREQUAL STD)
    MESSAGE("Compile with A* standard implementation")
    ADD_DEFINITIONS(-DASTAR_STD_IMPL)
elseif (ASTAR_IMPL STREQUAL IDA)
    MESSAGE("Compile with IDA* implementation")
    ADD_DEFINITIONS(-DIDASTAR_IMPL)
elseif(ASTAR_IMPL STREQUAL IDAOPT)
    MESSAGE("Compile with IDA* optimized implementation")
    ADD_DEFINITIONS(-DIDASTAR_OPT_IMPL)
else ()
    MESSAGE("Compile with A* standard implementation")
    ADD_DEFINITIONS(-DASTAR_STD_IMPL)
endif ()

set(DATASET_SCIENTIFIC_PRECISION 8)
MESSAGE("Setting scientific precision to " ${DATASET_SCIENTIFIC_PRECISION})
ADD_DEFINITIONS(-DDATASET_SCIENTIFC_PRECISION=${DATASET_SCIENTIFIC_PRECISION})

find_package(MPI REQUIRED)
find_package(GSL REQUIRED)

set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_SOURCE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)
set(TEST_SOURCE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
set(INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/includes)

set(BOOST_INCLUDE_DIRECTORY "" CACHE String "Boost include directory (local)")
include_directories(${BOOST_INCLUDE_DIRECTORY} /usr/local/include)
set(BOOST_LIBRARY_DIRECTORY "" CACHE String "Boost library directory (local)")
link_directories(${BOOST_LIBRARY_DIRECTORY} /usr/local/lib/libmlpack.so.3.0)

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
        includes/spatial_elements.hpp includes/astar.hpp)
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
        includes/runners/box_runner.hpp
        includes/report.hpp
        includes/graph.hpp
        includes/zoltan_fn.hpp
        includes/metrics.hpp includes/runners/simulator.hpp includes/runners/dataset_generator.hpp)

set_target_properties(build 
        PROPERTIES
        OUTPUT_NAME "nbmpi")

set(ZOLTAN_LIBRARY_PATH zoltan CACHE String "zoltan library path (zoltan.a)")
set(ARMADILLO_LIBRARY_PATH armadillo CACHE String "armadillo library path (libarmadillo.so)")
set(MLPACK_LIBRARY_PATH mlpack CACHE String "armadillo library path (libmlpack.so)")

target_link_libraries(build
        PRIVATE
        ${MPI_C_LIBRARIES}
        ${ZOLTAN_LIBRARY_PATH}
        ${GSL_LIBRARIES})

set(MLPACK_INCLUDE_DIRECTORY "" CACHE String "MLpack include directory")
set(LIBRL_INCLUDE_DIRECTORY  "" CACHE String "librl include directory")
set(ARMADILLO_INCLUDE_DIRECTORY "" CACHE String "armadillo include directory")
set(ZOLTAN_INCLUDE_DIRECTORY "" CACHE String "zoltan include directory")

find_library(ARMADILLO
        NAMES armadillo libarmadillo
        HINTS "${ARMADILLO_LIBRARY_PATH}" /usr/local ~/local/lib
)

target_include_directories(build
        PRIVATE
        ${MPI_C_INCLUDE_PATH}
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
        includes/runners/box_runner.hpp
        includes/report.hpp
        includes/graph.hpp
        includes/zoltan_fn.hpp
        includes/metrics.hpp
        src/dataset_builder.cpp includes/runners/simulator.hpp includes/runners/dataset_generator.hpp)
target_include_directories(dataset
        PRIVATE
        ${MPI_C_INCLUDE_PATH}
        ${ZOLTAN_INCLUDE_DIRECTORY})
target_link_libraries(dataset
        PRIVATE
        ${MPI_C_LIBRARIES}
        ${ZOLTAN_LIBRARY_PATH}
        ${GSL_LIBRARIES}
        )

add_executable(full_dataset
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
        includes/runners/box_runner.hpp
        includes/report.hpp
        includes/graph.hpp
        includes/zoltan_fn.hpp
        includes/metrics.hpp
        src/full_dataset_builder.cpp includes/runners/simulator.hpp includes/runners/dataset_generator.hpp)
target_include_directories(full_dataset
        PRIVATE
        ${MPI_C_INCLUDE_PATH}
        ${ZOLTAN_INCLUDE_DIRECTORY})
target_link_libraries(full_dataset
        PRIVATE
        ${MPI_C_LIBRARIES}
        ${ZOLTAN_LIBRARY_PATH}
        ${GSL_LIBRARIES})

add_executable(astar
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
        includes/runners/box_runner.hpp
        includes/report.hpp
        includes/graph.hpp
        includes/zoltan_fn.hpp
        includes/metrics.hpp
        src/astar_search.cpp includes/initial_conditions.hpp includes/runners/simulator.hpp includes/runners/dataset_generator.hpp includes/runners/branch_and_bound.hpp includes/decision_makers/strategy.hpp includes/feature_container.hpp src/zoltan_migration.cpp)
target_include_directories(astar
        PRIVATE
        ${MPI_C_INCLUDE_PATH}
        ${ZOLTAN_INCLUDE_DIRECTORY})
target_link_libraries(astar
        PRIVATE
        ${MPI_C_LIBRARIES}
        ${ZOLTAN_LIBRARY_PATH}
        ${GSL_LIBRARIES})

add_executable(basegain
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
        includes/runners/box_runner.hpp
        includes/report.hpp
        includes/graph.hpp
        includes/zoltan_fn.hpp
        includes/metrics.hpp
        src/dataset_base_gain.cpp includes/runners/simulator.hpp includes/runners/dataset_generator.hpp)
target_include_directories(basegain
        PRIVATE
        ${MPI_C_INCLUDE_PATH}
        ${ZOLTAN_INCLUDE_DIRECTORY})
target_link_libraries(basegain
        PRIVATE
        ${MPI_C_LIBRARIES}
        ${ZOLTAN_LIBRARY_PATH}
        ${GSL_LIBRARIES})

add_executable(doall
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
        includes/runners/box_runner.hpp
        includes/report.hpp
        includes/graph.hpp
        includes/zoltan_fn.hpp
        includes/metrics.hpp
        includes/runners/simulator.hpp
        includes/runners/dataset_generator.hpp
        src/do_all.cpp includes/feature_container.hpp includes/generate.hpp)

target_include_directories(doall
        PRIVATE
        ${MPI_C_INCLUDE_PATH}
        ${MLPACK_INCLUDE_DIRECTORY}
        ${ARMADILLO_INCLUDE_DIRECTORY}
        ${ZOLTAN_INCLUDE_DIRECTORY}
        )

target_link_libraries(doall
        PRIVATE
        ${MPI_C_LIBRARIES}
        ${ZOLTAN_LIBRARY_PATH}
        ${MLPACK_LIBRARY_PATH}
        ${ARMADILLO_LIBRARY_PATH}
        ${ARMADILLO}
        ${GSL_LIBRARIES})

add_executable(zoltan_migration
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
        includes/runners/box_runner.hpp
        includes/report.hpp
        includes/graph.hpp
        includes/zoltan_fn.hpp
        includes/metrics.hpp
        includes/runners/simulator.hpp
        includes/runners/dataset_generator.hpp
        src/zoltan_migration.cpp includes/feature_container.hpp
        )

add_executable(generate
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
        includes/runners/box_runner.hpp
        includes/report.hpp
        includes/graph.hpp
        includes/zoltan_fn.hpp
        includes/metrics.hpp
        includes/runners/simulator.hpp
        includes/runners/dataset_generator.hpp
        src/generate.cpp includes/feature_container.hpp includes/generate.hpp)

target_include_directories(generate
        PRIVATE
        ${MPI_C_INCLUDE_PATH}
        ${MLPACK_INCLUDE_DIRECTORY}
        ${ARMADILLO_INCLUDE_DIRECTORY}
        ${ZOLTAN_INCLUDE_DIRECTORY}
        )

target_link_libraries(generate
        PRIVATE
        ${MPI_C_LIBRARIES}
        ${ZOLTAN_LIBRARY_PATH}
        ${MLPACK_LIBRARY_PATH}
        ${ARMADILLO_LIBRARY_PATH}
        ${ARMADILLO}
        ${GSL_LIBRARIES})

target_include_directories(zoltan_migration
        PRIVATE
        ${MPI_C_INCLUDE_PATH}
        ${MLPACK_INCLUDE_DIRECTORY}
        ${ARMADILLO_INCLUDE_DIRECTORY}
        ${ZOLTAN_INCLUDE_DIRECTORY}
        )

target_link_libraries(zoltan_migration
        PRIVATE
        ${MPI_C_LIBRARIES}
        ${ZOLTAN_LIBRARY_PATH}
        ${MLPACK_LIBRARY_PATH}
        ${ARMADILLO_LIBRARY_PATH}
        ${ARMADILLO}
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

