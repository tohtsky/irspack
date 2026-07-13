cmake_minimum_required(VERSION 3.15...3.27)
project(irspack VERSION 0.1.0)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()


if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/eigen-3.4.0")
  file(DOWNLOAD https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip "${CMAKE_CURRENT_BINARY_DIR}/eigen-3.4.0.zip")
  file(ARCHIVE_EXTRACT INPUT "${CMAKE_CURRENT_BINARY_DIR}/eigen-3.4.0.zip")
endif()

find_package(Threads REQUIRED)


include_directories("${CMAKE_BINARY_DIR}/eigen-3.4.0" cpp_source)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


# Detect the installed nanobind package and import it into CMake
find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} ${SKBUILD_SABI_COMPONENT} REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_ials_core STABLE_ABI cpp_source/als/wrapper.cpp)
nanobind_add_module(_knn STABLE_ABI cpp_source/knn/wrapper.cpp)
nanobind_add_module(_util_cpp STABLE_ABI cpp_source/util.cpp)
nanobind_add_module(_core_evaluator STABLE_ABI cpp_source/evaluator.cpp)

install(TARGETS _ials_core LIBRARY DESTINATION irspack/recommenders)
install(TARGETS _core_evaluator LIBRARY DESTINATION irspack/evaluation)
install(TARGETS _util_cpp LIBRARY DESTINATION irspack/utils)
install(TARGETS _knn LIBRARY DESTINATION irspack/recommenders)

# Keep type stubs in lockstep with the extension modules.  INSTALL_TIME is
# necessary because the modules become importable only after scikit-build has
# assembled the Python package in the wheel staging directory.
function(add_extension_stub target module output)
  nanobind_add_stub(
    ${target}_stub
    INSTALL_TIME
    MODULE ${module}
    OUTPUT ${output}
    PYTHON_PATH "."
  )
endfunction()

add_extension_stub(_ials_core irspack.recommenders._ials_core
                   irspack/recommenders/_ials_core.pyi)
add_extension_stub(_core_evaluator irspack.evaluation._core_evaluator
                   irspack/evaluation/_core_evaluator.pyi)
add_extension_stub(_util_cpp irspack.utils._util_cpp irspack/utils/_util_cpp.pyi)
add_extension_stub(_knn irspack.recommenders._knn irspack/recommenders/_knn.pyi)
