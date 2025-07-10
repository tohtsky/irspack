#include "similarities.hpp"
#include <Eigen/Sparse>
#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>

using Real = double;

NB_MODULE(_knn, m) {
  nanobind::class_<KNN::CosineSimilarityComputer<Real>>(m, "CosineSimilarityComputer")
      .def(nanobind::init<const KNN::CosineSimilarityComputer<Real>::CSRMatrix &,
                    Real, bool, size_t, size_t>(),
           nanobind::arg("X"), nanobind::arg("shrinkage"), nanobind::arg("normalize"),
           nanobind::arg("n_threads") = 1, nanobind::arg("max_chunk_size") = 128)
      .def("compute_similarity",
           &KNN::CosineSimilarityComputer<Real>::compute_similarity);

  nanobind::class_<KNN::JaccardSimilarityComputer<Real>>(m,
                                                   "JaccardSimilarityComputer")
      .def(nanobind::init<const KNN::JaccardSimilarityComputer<Real>::CSRMatrix &,
                    Real, size_t, size_t>(),
           nanobind::arg("X"), nanobind::arg("shrinkage"), nanobind::arg("n_threads") = 1,
           nanobind::arg("max_chunk_size") = 128)
      .def("compute_similarity",
           &KNN::JaccardSimilarityComputer<Real>::compute_similarity);

  nanobind::class_<KNN::TverskyIndexComputer<Real>>(m, "TverskyIndexComputer")
      .def(nanobind::init<const KNN::TverskyIndexComputer<Real>::CSRMatrix &, Real,
                    Real, Real, size_t, size_t>(),
           nanobind::arg("X"), nanobind::arg("shrinkage"), nanobind::arg("alpha"),
           nanobind::arg("beta"), nanobind::arg("n_threads") = 1,
           nanobind::arg("max_chunk_size") = 128)
      .def("compute_similarity",
           &KNN::TverskyIndexComputer<Real>::compute_similarity);

  nanobind::class_<KNN::AsymmetricCosineSimilarityComputer<Real>>(
      m, "AsymmetricSimilarityComputer")
      .def(nanobind::init<
               const KNN::AsymmetricCosineSimilarityComputer<Real>::CSRMatrix &,
               Real, Real, size_t, size_t>(),
           nanobind::arg("X"), nanobind::arg("shrinkage"), nanobind::arg("alpha"),
           nanobind::arg("n_threads") = 1, nanobind::arg("max_chunk_size") = 128)
      .def("compute_similarity",
           &KNN::AsymmetricCosineSimilarityComputer<Real>::compute_similarity);

  nanobind::class_<KNN::P3alphaComputer<Real>>(m, "P3alphaComputer")
      .def(nanobind::init<const KNN::P3alphaComputer<Real>::CSRMatrix &, Real, size_t,
                    size_t>(),
           nanobind::arg("X"), nanobind::arg("alpha") = 0, nanobind::arg("n_threads") = 1,
           nanobind::arg("max_chunk_size") = 128)
      .def("compute_W", &KNN::P3alphaComputer<Real>::compute_W);

  nanobind::class_<KNN::RP3betaComputer<Real>>(m, "RP3betaComputer")
      .def(nanobind::init<const KNN::RP3betaComputer<Real>::CSRMatrix &, Real, Real,
                    size_t, size_t>(),
           nanobind::arg("X"), nanobind::arg("alpha") = 0, nanobind::arg("beta") = 0,
           nanobind::arg("n_threads") = 1, nanobind::arg("max_chunk_size") = 128)
      .def("compute_W", &KNN::RP3betaComputer<Real>::compute_W);
}
