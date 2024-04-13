cmake_minimum_required(VERSION 3.27)  # Ensure compatibility

project(DustRayTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Add subdirectories for dependencies early so they're available to link against
add_subdirectory(${PROJECT_NAME}/thirdparty/glad)
add_subdirectory(${PROJECT_NAME}/thirdparty/glfw)
add_subdirectory(${PROJECT_NAME}/thirdparty/imgui)


get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (is_multi_config)
    # Compute the bitness of the target, same as VS's $(PlatformArchitecture)
    math(EXPR PlatformArchitecture "8 * ${CMAKE_SIZEOF_VOID_P}")
    
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY 
    "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}-${PlatformArchitecture}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY 
    "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}-${PlatformArchitecture}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY 
    "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}-${PlatformArchitecture}")
endif ()

# List of source files
set(SOURCE_FILES 
./DustRayTracer/src/core/Application/private/Application.cpp
./DustRayTracer/src/core/Main/main.cpp
./DustRayTracer/thirdparty/stb_image/stb_image_write.cpp
./DustRayTracer/src/core/Editor/EditorLayer.cu
./DustRayTracer/src/core/Renderer/private/Renderer.cu
./DustRayTracer/src/core/Renderer/private/Camera/Camera.cu
./DustRayTracer/src/core/Renderer/private/Kernel/RenderKernel.cu
)

# Define the executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

add_dependencies(${PROJECT_NAME} glfw)

set_target_properties(${PROJECT_NAME}
PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Set the include directories
target_include_directories(${PROJECT_NAME} 
PUBLIC ${PROJECT_SOURCE_DIR}/DustRayTracer/src 
${PROJECT_SOURCE_DIR}/DustRayTracer/thirdparty/stb_image
${PROJECT_SOURCE_DIR}/DustRayTracer/thirdparty/glfw/include
${PROJECT_SOURCE_DIR}/DustRayTracer/thirdparty/glm
)

# Compiler options for CUDA
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -arch=compute_75 -code=sm_75 --expt-relaxed-constexpr --extended-lambda")

find_package(CUDAToolkit)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       -arch=compute_75 
                       -code=sm_75
                       --expt-relaxed-constexpr 
                       --extended-lambda
                       >)



# Link libraries
target_link_directories(${PROJECT_NAME} PRIVATE ${CMAKE_BINARY_DIR}/DustRayTracer/thirdparty/glfw/src/Debug)
target_link_libraries(${PROJECT_NAME} GLAD)
target_link_libraries(${PROJECT_NAME} glfw3)
target_link_libraries(${PROJECT_NAME} imgui)
target_link_libraries(${PROJECT_NAME} cudart)

# Set this project as the startup project in Visual Studio
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
