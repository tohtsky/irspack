#include "knn.hpp"
#include "similarities.hpp"
#include <Eigen/Sparse>
#include <cstddef>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;
using Real = double;

PYBIND11_MODULE(_knn, m) {
  py::class_<KNN::CosineSimilarityComputer<Real>>(m, "CosineSimilarityComputer")
      .def(py::init<const KNN::CosineSimilarityComputer<Real>::CSRMatrix &,
                    Real, bool, size_t, size_t>(),
           py::arg("X"), py::arg("shrinkage"), py::arg("normalize"),
           py::arg("n_threads") = 1, py::arg("max_chunk_size") = 128)
      .def("compute_similarity",
           &KNN::CosineSimilarityComputer<Real>::compute_similarity);

  py::class_<KNN::JaccardSimilarityComputer<Real>>(m,
                                                   "JaccardSimilarityComputer")
      .def(py::init<const KNN::JaccardSimilarityComputer<Real>::CSRMatrix &,
                    Real, size_t, size_t>(),
           py::arg("X"), py::arg("shrinkage"), py::arg("n_threads") = 1,
           py::arg("max_chunk_size") = 128)
      .def("compute_similarity",
           &KNN::JaccardSimilarityComputer<Real>::compute_similarity);

  py::class_<KNN::TverskyIndexComputer<Real>>(m, "TverskyIndexComputer")
      .def(py::init<const KNN::TverskyIndexComputer<Real>::CSRMatrix &, Real,
                    Real, Real, size_t, size_t>(),
           py::arg("X"), py::arg("shrinkage"), py::arg("alpha"),
           py::arg("beta"), py::arg("n_threads") = 1,
           py::arg("max_chunk_size") = 128)
      .def("compute_similarity",
           &KNN::TverskyIndexComputer<Real>::compute_similarity);

  py::class_<KNN::AsymmetricCosineSimilarityComputer<Real>>(
      m, "AsymmetricSimilarityComputer")
      .def(py::init<
               const KNN::AsymmetricCosineSimilarityComputer<Real>::CSRMatrix &,
               Real, Real, size_t, size_t>(),
           py::arg("X"), py::arg("shrinkage"), py::arg("alpha"),
           py::arg("n_threads") = 1, py::arg("max_chunk_size") = 128)
      .def("compute_similarity",
           &KNN::AsymmetricCosineSimilarityComputer<Real>::compute_similarity);

  py::class_<KNN::P3alphaComputer<Real>>(m, "P3alphaComputer")
      .def(py::init<const KNN::P3alphaComputer<Real>::CSRMatrix &, Real, size_t,
                    size_t>(),
           py::arg("X"), py::arg("alpha") = 0, py::arg("n_threads") = 1,
           py::arg("max_chunk_size") = 128)
      .def("compute_W", &KNN::P3alphaComputer<Real>::compute_W);

  py::class_<KNN::RP3betaComputer<Real>>(m, "RP3betaComputer")
      .def(py::init<const KNN::RP3betaComputer<Real>::CSRMatrix &, Real, Real,
                    size_t, size_t>(),
           py::arg("X"), py::arg("alpha") = 0, py::arg("beta") = 0,
           py::arg("n_threads") = 1, py::arg("max_chunk_size") = 128)
      .def("compute_W", &KNN::RP3betaComputer<Real>::compute_W);
}
