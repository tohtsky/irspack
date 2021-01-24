#pragma once
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "argcheck.hpp"

namespace irspack {
namespace sparse_util {

template <typename Real>
using CSRMatrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;

template <typename Real>
using RowMajorMatrix =
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename Real>
using ColMajorMatrix =
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template <typename Real>
using DenseVector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

template <typename Real>
using DenseColVector = Eigen::Matrix<Real, 1, Eigen::Dynamic, Eigen::RowMajor>;

template <typename Real>
using CSCMatrix = Eigen::SparseMatrix<Real, Eigen::ColMajor>;

template <typename Real>
inline CSRMatrix<Real> parallel_sparse_product(const CSRMatrix<Real> &left,
                                               const CSCMatrix<Real> &right,
                                               const size_t n_threads) {
  CSRMatrix<Real> result(left.rows(), right.cols());
  check_arg(n_threads > 0, "n_thraed must be > 0");
  const int n_row = left.rows();
  const int rows_per_block = n_row / n_threads;
  const int remnant = n_row % n_threads;
  int start = 0;
  std::vector<std::future<CSRMatrix<Real>>> workers;
  for (int i = 0; i < static_cast<int>(n_threads); i++) {
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
  for (int i = 0; i < static_cast<int>(n_threads); i++) {
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
std::pair<CSRMatrix<Real>, CSRMatrix<Real>>
train_test_split_rowwise(const CSRMatrix<Real> &X, const double test_ratio,
                         std::int64_t random_seed) {
  using Triplet = Eigen::Triplet<Integer>;
  std::mt19937 random_state(random_seed);
  check_arg(((test_ratio <= 1.0 && (test_ratio >= 0.0))),
            "test_ratio must be within [0.0, 1.0]");
  std::vector<Integer> col_buffer;
  std::vector<Real> data_buffer;
  std::vector<uint64_t> index_;
  std::vector<Triplet> train_data, test_data;
  for (int row = 0; row < X.outerSize(); ++row) {
    col_buffer.clear(); // does not change capacity
    data_buffer.clear();
    index_.clear();
    Integer cnt = 0;
    for (typename CSRMatrix<Real>::InnerIterator it(X, row); it; ++it) {
      index_.push_back(cnt);
      col_buffer.push_back(it.col());
      data_buffer.push_back(it.value());
      cnt += 1;
    }
    std::shuffle(index_.begin(), index_.end(), random_state);
    size_t n_test = static_cast<Integer>(std::floor(cnt * test_ratio));
    for (size_t i = 0; i < n_test; i++) {
      test_data.emplace_back(row, col_buffer[index_[i]],
                             data_buffer[index_[i]]);
    }
    for (size_t i = n_test; i < col_buffer.size(); i++) {
      train_data.emplace_back(row, col_buffer[index_[i]],
                              data_buffer[index_[i]]);
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

template <typename Real>
CSRMatrix<Real> okapi_BM_25_weight(const CSRMatrix<Real> &X, Real k1, Real b) {
  CSRMatrix<Real> result(X);
  using itertype = typename CSRMatrix<Real>::InnerIterator;
  const int N = X.rows();
  result.makeCompressed();
  DenseVector<Real> idf(X.cols());
  DenseVector<Real> doc_length(N);
  idf.array() = 0;
  doc_length.array() = 0;

  for (int i = 0; i < N; i++) {
    for (itertype iter(X, i); iter; ++iter) {
      idf(iter.col()) += 1;
      doc_length(i) += iter.value();
    }
  }
  Real avgdl = doc_length.sum() / N;
  idf.array() =
      (N / (idf.array() + static_cast<Real>(1)) + static_cast<Real>(1)).log();
  for (int i = 0; i < N; i++) {
    Real regularizer = k1 * (1 - b + b * doc_length(i) / avgdl);
    for (itertype iter(result, i); iter; ++iter) {
      iter.valueRef() = idf(iter.col()) * (iter.valueRef() * (k1 + 1)) /
                        (iter.valueRef() + regularizer);
    }
  }
  return result;
}

template <typename Real>
CSRMatrix<Real> tf_idf_weight(const CSRMatrix<Real> &X, bool smooth) {
  CSRMatrix<Real> result(X);
  using itertype = typename CSRMatrix<Real>::InnerIterator;
  const int N = X.rows();
  result.makeCompressed();
  DenseVector<Real> idf(X.cols());
  idf.array() = 0;

  for (int i = 0; i < N; i++) {
    for (itertype iter(X, i); iter; ++iter) {
      idf(iter.col()) += 1;
    }
  }
  idf.array() = (N / (idf.array() + static_cast<Real>(smooth))).log();
  for (int i = 0; i < N; i++) {
    for (itertype iter(result, i); iter; ++iter) {
      iter.valueRef() *= idf(iter.col());
    }
  }
  return result;
}

template <typename Real>
CSRMatrix<Real> remove_diagonal(const CSRMatrix<Real> &X) {
  check_arg(X.rows() == X.cols(), "X must be square");
  CSRMatrix<Real> result(X);
  using itertype = typename CSRMatrix<Real>::InnerIterator;
  const int N = X.rows();
  result.makeCompressed();
  for (int i = 0; i < N; i++) {
    for (itertype iter(result, i); iter; ++iter) {
      if (i == iter.col()) {
        iter.valueRef() = static_cast<Real>(0);
      }
    }
  }
  return result;
}

template <typename Real, bool positive_only = false,
          int block_size = Eigen::internal::packet_traits<Real>::size>
inline CSCMatrix<Real> SLIM(const CSRMatrix<Real> &X, size_t n_threads,
                            size_t n_iter, Real l2_coeff, Real l1_coeff,
                            Real tol) {
  check_arg(n_threads > 0, "n_threads must be > 0.");
  check_arg(n_iter > 0, "n_iter must be > 0.");
  check_arg(l2_coeff >= 0, "l2_coeff must be > 0.");
  check_arg(l1_coeff >= 0, "l1_coeff must be > 0.");
  using MatrixType =
      Eigen::Matrix<Real, block_size, Eigen::Dynamic, Eigen::ColMajor>;
  using VectorType = Eigen::Matrix<Real, block_size, 1>;

  // CSRMatrix<Real> X_csr(X);
  CSCMatrix<Real> X_csc(X);
  X_csc.makeCompressed();
  using TripletType = Eigen::Triplet<Real>;
  using CSCIter = typename CSCMatrix<Real>::InnerIterator;
  std::vector<std::future<std::vector<TripletType>>> workers;
  std::atomic<int64_t> cursor(0);
  for (size_t th = 0; th < n_threads; th++) {
    workers.emplace_back(std::async(std::launch::async, [&cursor, &X_csc,
                                                         l2_coeff, l1_coeff,
                                                         n_iter, tol] {
      const int64_t F = X_csc.cols();
      std::mt19937 gen(0);
      std::vector<int64_t> indices(F);
      for (int64_t i = 0; i < F; i++) {
        indices[i] = i;
      }
      MatrixType remnants(block_size, X_csc.rows());
      MatrixType coeffs(block_size, F);
      VectorType coeff_temp(block_size);
      VectorType diff(block_size);
      VectorType linear(block_size);
      VectorType linear_plus(block_size);
      VectorType linear_minus(block_size);

      std::vector<TripletType> local_resuts;
      while (true) {
        int64_t current_cursor = cursor.fetch_add(block_size);
        if (current_cursor >= F) {
          break;
        }

        int64_t block_begin = current_cursor;
        int64_t block_end = std::min(block_begin + block_size, F);
        int64_t valid_block_size = block_end - block_begin;
        remnants.setZero();
        coeffs.setZero();

        for (int64_t f_cursor = block_begin; f_cursor < block_end; f_cursor++) {
          const int64_t internal_col_position = f_cursor - block_begin;
          for (CSCIter iter(X_csc, f_cursor); iter; ++iter) {
            remnants(internal_col_position, iter.row()) = -iter.value();
          }
        }

        for (size_t cd_iteration = 0; cd_iteration < n_iter; cd_iteration++) {
          std::shuffle(indices.begin(), indices.end(), gen);
          Real delta = 0;
          for (int64_t feature_index = 0; feature_index < F; feature_index++) {
            int64_t shuffled_feature_index = indices[feature_index];
            coeff_temp = coeffs.col(shuffled_feature_index);
            diff = coeff_temp;
            linear.setZero();
            Real x2_sum = 0.0;
            for (CSCIter nnz_iter(X_csc, shuffled_feature_index); nnz_iter;
                 ++nnz_iter) {
              Real x = nnz_iter.value();

              const int64_t row = nnz_iter.row();
              x2_sum += x * x;
              /*
              loss = \sum_u (remnant_u - w^old _f X_uf + w^new _f X_uf ) ^2
              z_new =
              CONST
              + \sum_u X_{uf} ^2 w^new_f ^2
              + 2 * w^new_f \sum_u X_{uf} ( remnant_u - X_{uf} w^{old}_f )

              LINEAR_COEFF =
                \sum_u X_{uf} ( remnant_u ) -
                - \sum _u ( X_{uf} ^2) w^{old}_f

              */

              // remnants.col(row).noalias() -= x * coeff_temp;
              linear.noalias() += x * remnants.col(row);
            }
            linear.noalias() -= x2_sum * coeff_temp;

            Real quadratic = x2_sum + l2_coeff;
            linear_plus.array() = (-linear.array() - l1_coeff) / quadratic;
            if (!positive_only) {
              linear_minus.array() = (-linear.array() + l1_coeff) / quadratic;
            }
            // linear_plus /= quadratic;

            Real *ptr_location =
                coeffs.data() + shuffled_feature_index * block_size;
            Real *lp_ptr = linear_plus.data();
            Real *lm_ptr = linear_minus.data();

            for (int64_t inner_cursor_position = 0;
                 inner_cursor_position < block_size; inner_cursor_position++) {
              Real lplus = *(lp_ptr++);
              Real lminus = *(lm_ptr++);
              int64_t original_cursor_position =
                  inner_cursor_position + block_begin;
              if (original_cursor_position == shuffled_feature_index) {
                *(ptr_location++) = 0.0;
                continue;
              }
              if (positive_only) {
                if (lplus > 0) {
                  *(ptr_location++) = lplus;
                } else {
                  *(ptr_location++) = static_cast<Real>(0.0);
                }

              } else {
                if (lplus > 0) {
                  *(ptr_location++) = lplus;
                } else {
                  if (lminus < 0) {
                    *(ptr_location++) = lminus;
                  } else {
                    *(ptr_location++) = static_cast<Real>(0.0);
                  }
                }
              } // allow nagative block
            }
            coeff_temp.noalias() =
                coeffs.col(shuffled_feature_index) - coeff_temp;

            if (!coeff_temp.isZero()) {
              for (CSCIter nnz_iter(X_csc, shuffled_feature_index); nnz_iter;
                   ++nnz_iter) {
                const int64_t row = nnz_iter.row();
                remnants.col(row).noalias() += nnz_iter.valueRef() * coeff_temp;
              }
              delta = std::max(delta, coeff_temp.cwiseAbs().array().maxCoeff());
            }
          }
          if (delta < tol) {
            break;
          }
        }

        for (int64_t f = 0; f < F; f++) {
          for (int64_t inner_cursor_position = 0;
               inner_cursor_position < valid_block_size;
               inner_cursor_position++) {
            int64_t original_location = inner_cursor_position + block_begin;
            Real c = coeffs(inner_cursor_position, f);
            if (c != 0.0) {
              local_resuts.emplace_back(f, original_location, c);
            }
          }
        }
      }
      return local_resuts;
    }));
  }
  std::vector<TripletType> nnzs;
  for (auto &fres : workers) {
    auto result = fres.get();
    for (const auto &w : result) {
      nnzs.emplace_back(w);
    }
  }

  CSCMatrix<Real> result(X.cols(), X.cols());
  result.setFromTriplets(nnzs.begin(), nnzs.end());
  result.makeCompressed();
  return result;
}

} // namespace sparse_util
} // namespace irspack
