cmake_minimum_required(VERSION 3.15)
project(nbmpi)

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if("${CMAKE_BUILD_TYPE}" STREQUAL Sanitize)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
    message("Flags for build type \"${CMAKE_BUILD_TYPE}\" are: ${CMAKE_CXX_FLAGS}")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL Profile)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -p ${CMAKE_CXX_FLAGS_DEBUG}")
    message("Flags for build type \"${CMAKE_BUILD_TYPE}\" are: ${CMAKE_CXX_FLAGS}")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_LIST_DIR}/lib)
set(INCLUDE_DIRECTORY               ${CMAKE_CURRENT_LIST_DIR}/include)

find_package (Threads)

file(GLOB SRC_FILES src/*.cpp)
file(GLOB_RECURSE INCLUDE_FILES include/yalbb/*.hpp)

add_library(yalbb STATIC ${SRC_FILES} ${INCLUDE_FILES} zupply/src/zupply.cpp src/output_formatter.cpp src/coordinate_translater.cpp include/yalbb/load_balancing.hpp include/yalbb/experiment.hpp include/yalbb/run.hpp src/element.cpp include/yalbb/element.hpp)

if(MPI_COMPILE_FLAGS)
    set_target_properties(yalbb PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(yalbb PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

target_include_directories(yalbb PUBLIC  ${MPI_INCLUDE_PATH} include/yalbb zupply/src)

target_link_libraries(yalbb PUBLIC ${MPI_LIBRARIES})
target_link_libraries(yalbb PRIVATE ${CMAKE_THREAD_LIBS_INIT})
set_target_properties(yalbb PROPERTIES PUBLIC_HEADER "${INCLUDE_FILES}")

install(TARGETS yalbb
        ARCHIVE
          DESTINATION ${CMAKE_INSTALL_PREFIX}/yalbb/lib
          COMPONENT Libraries
        PUBLIC_HEADER
          DESTINATION ${CMAKE_INSTALL_PREFIX}/yalbb/include
        COMPONENT Development)

########################################################################################################################

