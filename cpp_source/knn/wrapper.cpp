/**
 * Copyright 2020 BizReach, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
  using CosineKNNComputer = CosineKNNComputer<double>;
  using CSRMatrix = typename CosineKNNComputer::CSRMatrix;
  py::class_<CosineKNNComputer>(m, "CosineKNNComputer")
      .def(py::init<const CSRMatrix &, size_t, double>())
      .def("compute_block", &CosineKNNComputer::compute_similarity);
}