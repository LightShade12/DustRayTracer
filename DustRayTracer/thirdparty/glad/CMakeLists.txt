cmake_minimum_required(VERSION 3.27)  #3.19

project(GLAD LANGUAGES C)

# Set output directories for library files
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Differentiate output directories for debug and release builds
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/lib/debug)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib/release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/lib/debug)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib/release)

# Add the GLAD library
add_library(${PROJECT_NAME} STATIC src/glad.c)

# Specify interface include directories
target_include_directories(GLAD PUBLIC ${PROJECT_SOURCE_DIR}/include)
