cmake_minimum_required(VERSION 3.0.0)
project(rs_evaluation VERSION 0.1.0)
set(PYBIND11_PYTHON_VERSION 3.7)

find_package(Threads REQUIRED)

include_directories(eigen-3.3.7 cpp_source)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(rs_evaluation main.cpp)
target_link_libraries(rs_evaluation Threads::Threads)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_subdirectory(pybind11)
pybind11_add_module(_ials cpp_source/als/wrapper.cpp)
