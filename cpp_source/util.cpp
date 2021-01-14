#include "util.hpp"
#include "pybind11/cast.h"
#include <cstddef>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using namespace irspack;
PYBIND11_MODULE(_util_cpp, m) {

  m.def("remove_diagonal", &sparse_util::remove_diagonal<double>);
  m.def("sparse_mm_threaded", &sparse_util::parallel_sparse_product<double>);
  m.def("rowwise_train_test_split_d",
        &sparse_util::train_test_split_rowwise<double>);
  m.def("rowwise_train_test_split_f",
        &sparse_util::train_test_split_rowwise<float>);
  m.def("rowwise_train_test_split_i",
        &sparse_util::train_test_split_rowwise<float>);
  m.def("okapi_BM_25_weight", &sparse_util::okapi_BM_25_weight<double>,
        py::arg("X"), py::arg("k1") = 1.2, py::arg("b") = 0.75);
  m.def("tf_idf_weight", &sparse_util::tf_idf_weight<double>, py::arg("X"),
        py::arg("smooth") = true);

  m.def("slim_weight_allow_negative", &sparse_util::SLIM<float, false>,
        py::arg("X"), py::arg("n_threads"), py::arg("n_iter"),
        py::arg("l2_coeff"), py::arg("l1_coeff"), py::arg("tol"));

  m.def("slim_weight_positive_only", &sparse_util::SLIM<float, true>,
        py::arg("X"), py::arg("n_threads"), py::arg("n_iter"),
        py::arg("l2_coeff"), py::arg("l1_coeff"), py::arg("tol"));
}
