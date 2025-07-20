cmake_minimum_required(VERSION 4.0)

project(CXX_GRAPHICS LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
find_package(Vulkan)
add_executable(example-vulkan src/Vulkan.cc)
target_link_libraries(example-vulkan Vulkan::Vulkan glfw)