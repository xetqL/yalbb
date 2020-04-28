cmake_minimum_required(VERSION 3.15)
project(nbmpi)

MESSAGE("Compile with A* standard implementation")
ADD_DEFINITIONS(-DASTAR_STD_IMPL)

find_package(MPI REQUIRED)

set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if("${CMAKE_BUILD_TYPE}" STREQUAL Sanitize)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
    message("Flags for build type \"${CMAKE_BUILD_TYPE}\" are: ${CMAKE_CXX_FLAGS}")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_SOURCE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)
set(TEST_SOURCE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
set(INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/includes)

find_library(ZOLTAN
        NAMES zoltan libzoltan
        HINTS /usr/local ~/local/lib ${CMAKE_PREFIX_PATH}/Zoltan/lib
        PATHS ${ZOLTAN_LIBRARY_PATH})
find_path(ZOLTAN_INCLUDES
        NAMES zoltan zoltan.h
        HINTS ${CMAKE_PREFIX_PATH}/Zoltan/include
        )

########################################################################################################################
# Executable
########################################################################################################################
## Main build

add_executable(build
        ${EXECUTABLE_SOURCE_DIRECTORY}/nbmpi.cpp
        ${INCLUDE_DIRECTORY}/utils.hpp
        ${INCLUDE_DIRECTORY}/parallel_utils.hpp
        includes/spatial_elements.hpp
        includes/ljpotential.hpp
        includes/physics.hpp
        includes/nbody_io.hpp
        includes/params.hpp
        includes/report.hpp
        includes/zoltan_fn.hpp
        includes/runners/simulator.hpp
        includes/communication_datatype.hpp
        includes/runners/shortest_path.hpp)

# speedlog build toolchain and link
execute_process(COMMAND mkdir -p build WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/spdlog)
execute_process(COMMAND cmake .. WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/spdlog/build)
execute_process(COMMAND make -j WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/spdlog/build)
target_link_libraries(build PRIVATE ${CMAKE_CURRENT_LIST_DIR}/spdlog/build/libspdlog.a)
target_include_directories(build PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/spdlog/include
)
# end of processing speedlog

set_target_properties(build 
        PROPERTIES
        OUTPUT_NAME "nbmpi")

set(ZOLTAN_LIBRARY_PATH zoltan CACHE STRING "zoltan library path (zoltan.a)")

target_link_libraries(build PRIVATE
        ${MPI_C_LIBRARIES}
        ${ZOLTAN_LIBRARY_PATH})

#set(ZOLTAN_INCLUDE_DIRECTORY "" CACHE STRING "zoltan include directory")

target_include_directories(build
        PRIVATE
        ${MPI_C_INCLUDE_PATH}
        ${ZOLTAN_INCLUDE_DIRECTORY}
        ${CMAKE_CURRENT_LIST_DIR}/zupply/src
)
########################################################################################################################

