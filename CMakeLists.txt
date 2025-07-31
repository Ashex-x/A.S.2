cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(VisionTransformer)

# Find the LibTorch package
set(CMAKE_PREFIX_PATH $ENV{HOME}/Dev/lib/libtorch)
find_package(Torch REQUIRED)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Check if CUDA is available and set flags accordingly
if(TORCH_CUDA_VERSION)
    message(STATUS "CUDA support is enabled in LibTorch.")
    add_definitions(-DUSE_CUDA)
else()
    message(STATUS "CUDA support is NOT enabled in LibTorch.")
endif()

set(EXECUTABLE_NAME "vit_classifier")

# Add the executable
add_executable(${EXECUTABLE_NAME} dynamo_/src/main.cpp)

# Link against Torch
target_link_libraries(${EXECUTABLE_NAME} "${TORCH_LIBRARIES}")

# Set C++ standard
set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
