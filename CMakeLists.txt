cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project (Abstract_Learning)

add_definitions(-DALSPEEDFIRST)

find_package(OpenCL)
if (OPENCL_FOUND)
    set(HAS_OCL TRUE)
    add_definitions(-DALOPENCL_MAC)
endif()


include_directories(include ./third ../Renascence/include/)
file(GLOB SRC_CPP "src/*.cpp" "src/*/*.cpp" "src/*/*/*.cpp" "src/*/*.c")
file(GLOB JSON "./third/cJSON/cJSON.c")
file(GLOB TEST_CPP "./test/*.cpp")
add_compile_options(-std=c++11)

add_library(Abstract_Learning SHARED ${SRC_CPP} ${JSON})

add_executable(example.out ./example.cpp)
add_executable(test.out ${TEST_CPP})
if (HAS_OCL)
    target_link_libraries(Abstract_Learning OpenCL)
endif()
target_link_libraries(example.out Abstract_Learning)
target_link_libraries(test.out Abstract_Learning)
