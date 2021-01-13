cmake_minimum_required(VERSION 3.0.0)
project(rs_evaluation VERSION 0.1.0)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -fPIC -O3")
set(PYBIND11_PYTHON_VERSION 3.7)

find_package(Threads REQUIRED)

include_directories(eigen-3.3.7 cpp_source)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_subdirectory(pybind11)
#pybind11_add_module(irspack.recommenders._ials cpp_source/als/wrapper.cpp)
#pybind11_add_module(irspack.recommenders._knn cpp_source/knn/wrapper.cpp)
pybind11_add_module(_util_cpp cpp_source/util.cpp)
#pybind11_add_module(irspack._evapuator cpp_source/evaluator.cpp)
#pybind11_add_module(irspack._rwr cpp_source/rws.cpp)
