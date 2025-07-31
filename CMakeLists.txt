cmake_minimum_required(VERSION 3.18)
project(torchtest)

set(CMAKE_PREFIX_PATH $ENV{HOME}/Dev/lib/libtorch)

find_package(Torch REQUIRED)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Executable
add_executable(${PROJECT_NAME} dynamo_/src/main.cpp)

# Link libraries.
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})

# C++ standard
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
