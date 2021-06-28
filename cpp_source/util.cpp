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
  m.def("rowwise_train_test_split_by_ratio",
        &sparse_util::SplitByRatioFunction<double>::split);
  m.def("rowwise_train_test_split_by_fixed_n",
        &sparse_util::SplitFixedN<double>::split);
  m.def("okapi_BM_25_weight", &sparse_util::okapi_BM_25_weight<double>,
        py::arg("X"), py::arg("k1") = 1.2, py::arg("b") = 0.75);
  m.def("tf_idf_weight", &sparse_util::tf_idf_weight<double>, py::arg("X"),
        py::arg("smooth") = true);

  m.def("slim_weight_allow_negative", &sparse_util::SLIM<float, false>,
        py::arg("X"), py::arg("n_threads"), py::arg("n_iter"),
        py::arg("l2_coeff"), py::arg("l1_coeff"), py::arg("tol"),
        py::arg("top_k") = -1);

  m.def("slim_weight_positive_only", &sparse_util::SLIM<float, true>,
        py::arg("X"), py::arg("n_threads"), py::arg("n_iter"),
        py::arg("l2_coeff"), py::arg("l1_coeff"), py::arg("tol"),
        py::arg("top_k") = -1);

  m.def("retrieve_recommend_from_score_f64",
        &sparse_util::retrieve_recommend_from_score<double>, py::arg("score"),
        py::arg("allowed_indices"), py::arg("cutoff"),
        py::arg("n_threads") = 1);
  m.def("retrieve_recommend_from_score_f32",
        &sparse_util::retrieve_recommend_from_score<float>, py::arg("score"),
        py::arg("allowed_indices"), py::arg("cutoff"),
        py::arg("n_threads") = 1);
}
