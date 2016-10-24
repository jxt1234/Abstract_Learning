cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project (Abstract_learning)

add_definitions(-DALSPEEDFIRST)

find_package(OpenCL)
if (OpenCL_FOUND)
    message(STATUS "OpenCL include : ${OpenCL_INCLUDE_DIR}")
    set(HAS_OCL TRUE)
    add_definitions(-DALOPENCL_MAC)
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

include_directories(include ./third ../Renascence/include/)
file(GLOB SRC_CPP "src/*.cpp" "src/*/*.cpp" "src/*/*/*.cpp" "src/*/*.c")
file(GLOB JSON "./third/cJSON/cJSON.c")
file(GLOB TEST_CPP "./test/*.cpp")

add_library(Abstract_learning SHARED ${SRC_CPP} ${JSON})

add_executable(example.out ./example.cpp)
add_executable(matrix_tool.out ./cpptools/matrix_tools.cpp)
add_executable(test.out ${TEST_CPP})
find_package (Threads)
if (HAS_OCL)
    target_link_libraries(Abstract_learning ${OpenCL_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
else()
    target_link_libraries(Abstract_learning ${CMAKE_THREAD_LIBS_INIT})
endif()
target_link_libraries(example.out Abstract_learning)
target_link_libraries(test.out Abstract_learning)
target_link_libraries(matrix_tool.out Abstract_learning)
