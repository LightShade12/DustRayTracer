cmake_minimum_required(VERSION 3.27)  #3.19

project(GLAD LANGUAGES C)

# Add the GLAD library
add_library(${PROJECT_NAME} STATIC src/glad.c)

# Specify interface include directories
target_include_directories(GLAD PUBLIC ${PROJECT_SOURCE_DIR}/include)
