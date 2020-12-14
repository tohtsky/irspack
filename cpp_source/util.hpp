#pragma once
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <future>
#include <random>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>

namespace sparse_util {

template <typename Real>
using CSRMatrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;

template <typename Real>
using CSCMatrix = Eigen::SparseMatrix<Real, Eigen::ColMajor>;

template <typename Real>
inline CSRMatrix<Real> parallel_sparse_product(const CSRMatrix<Real> &left,
                                               const CSCMatrix<Real> &right,
                                               const size_t n_thread) {
  CSRMatrix<Real> result(left.rows(), right.cols());
  if (n_thread == 0) {
    throw std::invalid_argument("n_thread must be > 0");
  }
  const int n_row = left.rows();
  const int rows_per_block = n_row / n_thread;
  const int remnant = n_row % n_thread;
  int start = 0;
  std::vector<std::future<CSRMatrix<Real>>> workers;
  for (int i = 0; i < static_cast<int>(n_thread); i++) {
    int block_size = rows_per_block;
    if (i < remnant) {
      ++block_size;
    }
    workers.emplace_back(std::async(std::launch::async, [&left, &right, start,
                                                         block_size]() {
      CSRMatrix<Real> local_result = left.middleRows(start, block_size) * right;
      return local_result;
    }));
    start += block_size;
  }
  start = 0;
  for (int i = 0; i < static_cast<int>(n_thread); i++) {
    int block_size = rows_per_block;
    if (i < remnant) {
      ++block_size;
    }
    result.middleRows(start, block_size) = workers[i].get();
    start += block_size;
  }
  return result;
}

template <typename Real, typename Integer = int64_t>
std::pair<CSCMatrix<Real>, CSRMatrix<Real>>
train_test_split_rowwise(const CSRMatrix<Real> &X, const double test_ratio,
                         std::int64_t random_seed) {
  using Triplet = Eigen::Triplet<Integer>;
  std::mt19937 random_state(random_seed);
  if (test_ratio > 1.0 || test_ratio < 0)
    throw std::invalid_argument("test_ratio must be within [0, 1]");
  std::vector<Integer> buffer;
  std::vector<Triplet> train_data, test_data;
  for (int row = 0; row < X.outerSize(); ++row) {
    buffer.clear(); // does not change capacity
    Integer cnt = 0;
    for (typename CSRMatrix<Real>::InnerIterator it(X, row); it; ++it) {
      cnt += it.value();
      buffer.push_back(it.col());
    }
    std::shuffle(buffer.begin(), buffer.end(), random_state);
    size_t n_test = static_cast<Integer>(std::floor(cnt * test_ratio));
    for (size_t i = 0; i < n_test; i++) {
      test_data.emplace_back(row, buffer[i], 1);
    }
    for (size_t i = n_test; i < buffer.size(); i++) {
      train_data.emplace_back(row, buffer[i], 1);
    }
  }
  CSRMatrix<Real> X_train(X.rows(), X.cols()), X_test(X.rows(), X.cols());
  auto dupfunction = [](const Integer &a, const Integer &b) { return a + b; };
  X_train.setFromTriplets(train_data.begin(), train_data.end(), dupfunction);
  X_test.setFromTriplets(test_data.begin(), test_data.end(), dupfunction);
  X_train.makeCompressed();
  X_test.makeCompressed();
  return {X_train, X_test};
}

} // namespace sparse_util
