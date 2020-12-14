#include "knn.hpp"
#include "pybind11/cast.h"
#include <Eigen/Sparse>
#include <cstddef>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

using namespace KNN;
namespace py = pybind11;
using std::vector;

PYBIND11_MODULE(_knn, m) {
  using KNNComputer = KNNComputer<double>;
  using CSRMatrix = typename KNNComputer::CSRMatrix;
  py::enum_<SimilarityType>(m, "SimilarityType");
  py::class_<KNNComputer>(m, "KNNComputer")
      .def(py::init<const CSRMatrix &, SimilarityType, size_t, double>())
      .def("compute_block", &KNNComputer::compute_similarity);
}