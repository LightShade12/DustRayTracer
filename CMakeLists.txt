cmake_minimum_required(VERSION 3.27)#3.19
project(DustRayTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(${PROJECT_NAME})

set(RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${PROJECT_SOURCE_DIR}/DustRayTracer/src)

add_subdirectory(DustRayTracer/thirdparty/glad)

target_link_libraries(${PROJECT_NAME} GLAD)