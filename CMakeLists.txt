cmake_minimum_required(VERSION 3.15)
project(nbmpi)

find_package(MPI REQUIRED)

set(CMAKE_CXX_COMPILER mpic++)
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

find_package (Threads)
file(GLOB SRC_FILES src/*.cpp)
file(GLOB_RECURSE INCLUDE_FILES includes/*.hpp)

add_library(yalbb STATIC ${SRC_FILES} ${INCLUDE_FILES} zupply/src/zupply.cpp)

target_include_directories(yalbb PRIVATE ${MPI_C_INCLUDE_PATH})
target_include_directories(yalbb PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/spdlog/include
        ${INCLUDE_DIRECTORY}
        zupply/src)

target_link_libraries(yalbb PRIVATE
        ${MPI_C_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT})

set_target_properties(yalbb PROPERTIES "PUBLIC_HEADER" "${INCLUDE_FILES}")

install(TARGETS yalbb
        LIBRARY
          DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
          COMPONENT Libraries
        PUBLIC_HEADER
          DESTINATION ${CMAKE_INSTALL_PREFIX}/include/yalbb
        COMPONENT Development)

########################################################################################################################

