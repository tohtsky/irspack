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