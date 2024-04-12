cmake_minimum_required(VERSION 3.27)  # Ensure compatibility

project(DustRayTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release)

# Add subdirectories for dependencies early so they're available to link against
add_subdirectory(thirdparty/glad)
add_subdirectory(thirdparty/glfw)
add_subdirectory(thirdparty/imgui)
add_subdirectory(thirdparty/glm)

# List of source files
set(SOURCE_FILES 
    ./src/core/Application/private/Application.cpp
    ./src/core/Main/main.cpp
    ./src/core/Renderer/private/main.cpp
    ./thirdparty/stb_image/stb_image.cpp
    ./src/core/Editor/EditorLayer.cu
    ./src/core/Renderer/private/Renderer.cu
    ./src/core/Renderer/private/Camera/Camera.cu
    ./src/core/Renderer/private/Kernel/RenderKernel.cu
)

# Define the executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Set the include directories
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/thirdparty/stb_image)

# Compiler options for CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -arch=compute_75 -code=sm_75 --expt-relaxed-constexpr --extended-lambda")

# Link libraries
target_link_libraries(${PROJECT_NAME} GLAD)
target_link_libraries(${PROJECT_NAME} GLFW)
target_link_libraries(${PROJECT_NAME} imgui)

# Set this project as the startup project in Visual Studio
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
