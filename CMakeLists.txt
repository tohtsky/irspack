cmake_minimum_required(VERSION 3.15...3.27)
project(irspack VERSION 0.1.0)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()



set(CMAKE_BUILD_TYPE DEBUG)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(Threads REQUIRED)

include_directories(eigen-3.4.0 cpp_source)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


# Detect the installed nanobind package and import it into CMake
find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)

nanobind_add_module(_ials_core cpp_source/als/wrapper.cpp)
nanobind_add_module(irspack.recommenders._knn cpp_source/knn/wrapper.cpp)
nanobind_add_module(irspack.utils._util_cpp cpp_source/util.cpp)
nanobind_add_module(_core_evaluator cpp_source/evaluator.cpp)

install(TARGETS _ials_core LIBRARY DESTINATION irspack/recommenders)
install(TARGETS _core_evaluator LIBRARY DESTINATION irspack/evaluation)

#nanobind_add_stub(
#  _ials_core_stub
#  MODULE irspack.recommenders._ials_core
#  OUTPUT irspack.recommenders._ials_core.pyi
#  PYTHON_PATH "./src"
#  DEPENDS _ials_core
#)