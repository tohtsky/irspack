#include "util.hpp"
#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>

using namespace irspack;
NB_MODULE(_util_cpp, m) {

  m.def("remove_diagonal", &sparse_util::remove_diagonal<double>);
  m.def("sparse_mm_threaded", &sparse_util::parallel_sparse_product<double>);
  m.def("rowwise_train_test_split_by_ratio",
        &sparse_util::SplitByRatioFunction<double>::split);
  m.def("rowwise_train_test_split_by_fixed_n",
        &sparse_util::SplitFixedN<double>::split);
  m.def("okapi_BM_25_weight", &sparse_util::okapi_BM_25_weight<double>,
        nanobind::arg("X"), nanobind::arg("k1") = 1.2, nanobind::arg("b") = 0.75);
  m.def("tf_idf_weight", &sparse_util::tf_idf_weight<double>, nanobind::arg("X"),
        nanobind::arg("smooth") = true);

  m.def("slim_weight_allow_negative", &sparse_util::SLIM<float, false>,
        nanobind::arg("X"), nanobind::arg("n_threads"), nanobind::arg("n_iter"),
        nanobind::arg("l2_coeff"), nanobind::arg("l1_coeff"), nanobind::arg("tol"),
        nanobind::arg("top_k") = -1);

  m.def("slim_weight_positive_only", &sparse_util::SLIM<float, true>,
        nanobind::arg("X"), nanobind::arg("n_threads"), nanobind::arg("n_iter"),
        nanobind::arg("l2_coeff"), nanobind::arg("l1_coeff"), nanobind::arg("tol"),
        nanobind::arg("top_k") = -1);

  m.def("retrieve_recommend_from_score_f64",
        &sparse_util::retrieve_recommend_from_score<double>, nanobind::arg("score"),
        nanobind::arg("allowed_indices"), nanobind::arg("cutoff"),
        nanobind::arg("n_threads") = 1);
  m.def("retrieve_recommend_from_score_f32",
        &sparse_util::retrieve_recommend_from_score<float>, nanobind::arg("score"),
        nanobind::arg("allowed_indices"), nanobind::arg("cutoff"),
        nanobind::arg("n_threads") = 1);
}
