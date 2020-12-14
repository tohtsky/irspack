#include "util.hpp"
#include "pybind11/cast.h"
#include <cstddef>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
PYBIND11_MODULE(_util_cpp, m) {

  m.def("sparse_mm_threaded", &sparse_util::parallel_sparse_product<double>);
  m.def("rowwise_train_test_split_d",
        &sparse_util::train_test_split_rowwise<double>);
  m.def("rowwise_train_test_split_f",
        &sparse_util::train_test_split_rowwise<float>);
  m.def("rowwise_train_test_split_i",
        &sparse_util::train_test_split_rowwise<float>);
}