#include "knn.hpp"
#include "pybind11/cast.h"
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
                    size_t, Real>(),
           py::arg("X"), py::arg("n_thread"), py::arg("shrinkage"))
      .def("compute_block",
           &KNN::CosineSimilarityComputer<Real>::compute_similarity);

  py::class_<KNN::JaccardSimilarityComputer<Real>>(m,
                                                   "JaccardSimilarityComputer")
      .def(py::init<const KNN::JaccardSimilarityComputer<Real>::CSRMatrix &,
                    size_t, Real>(),
           py::arg("X"), py::arg("n_thread"), py::arg("shrinkage"))
      .def("compute_block",
           &KNN::JaccardSimilarityComputer<Real>::compute_similarity);

  py::class_<KNN::AsymmetricCosineSimilarityComputer<Real>>(
      m, "AsymmetricSimilarityComputer")
      .def(py::init<
               const KNN::AsymmetricCosineSimilarityComputer<Real>::CSRMatrix &,
               size_t, Real, Real>(),
           py::arg("X"), py::arg("n_thread"), py::arg("shrinkage"),
           py::arg("alpha"))
      .def("compute_block",
           &KNN::AsymmetricCosineSimilarityComputer<Real>::compute_similarity);
}