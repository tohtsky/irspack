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
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

namespace ials11 {
using Real = float;
using IndexType = std::size_t;
using SparseMatrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using DenseMatrix =
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using DenseVector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
} // namespace ials11
