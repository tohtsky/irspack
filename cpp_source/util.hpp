#pragma once
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <tuple>
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
inline RowMajorMatrix<Real>
parallel_sparse_product(const CSRMatrix<Real> &left,
                        const CSCMatrix<Real> &right, const size_t n_threads) {
  RowMajorMatrix<Real> result(left.rows(), right.cols());
  result.array() = 0;
  check_arg(n_threads > 0, "n_thraed must be > 0");
  const int64_t n_row = left.rows();
  std::atomic<int64_t> cursor(0);
  std::vector<std::thread> workers;
  for (int i = 0; i < static_cast<int>(n_threads); i++) {
    workers.emplace_back([&left, &right, &cursor, n_row, &result]() {
      const int64_t chunk_size = 16;
      while (true) {
        auto current_position = cursor.fetch_add(chunk_size);
        if (current_position >= n_row) {
          break;
        }
        auto block_size =
            std::min(current_position + chunk_size, n_row) - current_position;
        result.middleRows(current_position, block_size) +=
            left.middleRows(current_position, block_size) * right;
      }
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
  return result;
}

template <typename Real, class Derived, typename Integer = int64_t>
struct SplitFunction {
  template <typename... Args>
  static std::pair<CSRMatrix<Real>, CSRMatrix<Real>>
  split_imple(const CSRMatrix<Real> &X, std::int64_t random_seed,
              Args... args) {
    Derived::check_args(args...);
    using Triplet = Eigen::Triplet<Integer>;
    std::mt19937 random_state(random_seed);
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
      size_t n_test = Derived::get_n_test(cnt, args...);
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
};

template <typename Real, typename Integer = int64_t>
struct SplitByRatioFunction
    : SplitFunction<Real, SplitByRatioFunction<Real, Integer>, Integer> {
  using Base =
      SplitFunction<Real, SplitByRatioFunction<Real, Integer>, Integer>;
  static void check_args(double test_ratio, bool n_test_ceil) {
    check_arg(((test_ratio <= 1.0 && (test_ratio >= 0.0))),
              "test_ratio must be within [0.0, 1.0]");
  }

  static size_t get_n_test(size_t nnz_row, double test_ratio,
                           bool n_test_ceil) {
    if (n_test_ceil) {
      return std::ceil(nnz_row * test_ratio);
    } else {
      return std::floor(nnz_row * test_ratio);
    }
  };

  static std::pair<CSRMatrix<Real>, CSRMatrix<Real>>
  split(const CSRMatrix<Real> &X, std::int64_t random_seed, Real heldout_ratio,
        bool n_test_ceil) {
    return Base::split_imple(X, random_seed, heldout_ratio, n_test_ceil);
  }
};

template <typename Real, typename Integer = int64_t>
struct SplitFixedN : SplitFunction<Real, SplitFixedN<Real, Integer>, Integer> {
  using Base = SplitFunction<Real, SplitFixedN<Real, Integer>, Integer>;
  static void check_args(size_t n_held_out) {}

  static size_t get_n_test(size_t nnz_row, size_t n_held_out) {
    return std::min(nnz_row, n_held_out);
  };

  static std::pair<CSRMatrix<Real>, CSRMatrix<Real>>
  split(const CSRMatrix<Real> &X, std::int64_t random_seed, size_t n_held_out) {
    return Base::split_imple(X, random_seed, n_held_out);
  }
};

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
                            Real tol, int64_t top_k) {
  check_arg(n_threads > 0, "n_threads must be > 0.");
  check_arg(n_iter > 0, "n_iter must be > 0.");
  check_arg(l2_coeff >= 0, "l2_coeff must be > 0.");
  check_arg(l1_coeff >= 0, "l1_coeff must be > 0.");
  using MatrixType =
      Eigen::Matrix<Real, block_size, Eigen::Dynamic, Eigen::ColMajor>;
  using VectorType = Eigen::Matrix<Real, block_size, 1>;

  CSCMatrix<Real> X_csc(X);
  X_csc.makeCompressed();
  using TripletType = Eigen::Triplet<Real>;
  using CSCIter = typename CSCMatrix<Real>::InnerIterator;
  using RealAndIndex = std::pair<Real, int64_t>;
  std::vector<std::future<std::vector<TripletType>>> workers;
  std::atomic<int64_t> cursor(0);
  for (size_t th = 0; th < n_threads; th++) {
    workers.emplace_back(std::async(std::launch::async, [&cursor, &X_csc,
                                                         l2_coeff, l1_coeff,
                                                         n_iter, tol, top_k] {
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

      std::vector<RealAndIndex> argsort_buffer;
      if (top_k >= 0) {
        argsort_buffer.resize(F);
      }

      std::vector<TripletType> local_results;
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
              x2_sum += x * x;
              linear.noalias() += x * remnants.col(nnz_iter.row());
            }
            linear.noalias() -= x2_sum * coeff_temp;

            Real quadratic = x2_sum + l2_coeff;
            linear_plus.array() = (-linear.array() - l1_coeff) / quadratic;
            if (!positive_only) {
              linear_minus.array() = (-linear.array() + l1_coeff) / quadratic;
            }

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

        if (top_k < 0) {
          for (int64_t f = 0; f < F; f++) {
            for (int64_t inner_cursor_position = 0;
                 inner_cursor_position < valid_block_size;
                 inner_cursor_position++) {
              int64_t original_location = inner_cursor_position + block_begin;
              Real c = coeffs(inner_cursor_position, f);
              if (c != 0.0) {
                local_results.emplace_back(f, original_location, c);
              }
            }
          }
        } else {
          for (int64_t inner_cursor_position = 0;
               inner_cursor_position < valid_block_size;
               inner_cursor_position++) {
            int64_t original_location = inner_cursor_position + block_begin;
            auto iter = argsort_buffer.begin();
            int64_t nnz = 0;
            for (int64_t f = 0; f < F; f++) {
              Real c = coeffs(inner_cursor_position, f);
              if (c != 0.0) {
                iter->first = c;
                iter->second = f;
                iter++;
                nnz++;
              }
            }
            int64_t n_taken_coeffs = nnz;
            if (nnz > top_k) {
              std::sort(argsort_buffer.begin(), argsort_buffer.end(),
                        [](RealAndIndex val1, RealAndIndex val2) {
                          return val1.first > val2.first;
                        });
              n_taken_coeffs = top_k;
            }
            for (int64_t i = 0; i < n_taken_coeffs; i++) {
              local_results.emplace_back(argsort_buffer[i].second,
                                         original_location,
                                         argsort_buffer[i].first);
            }
          }
        }
      }
      return local_results;
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

template <typename Real>
inline std::vector<std::vector<std::pair<int64_t, float>>>
retrieve_recommend_from_score(
    const RowMajorMatrix<Real> &score,
    const std::vector<std::vector<int64_t>> &allowed_indices,
    const size_t cutoff, size_t n_threads) {
  using score_and_index = std::pair<int64_t, float>;
  check_arg(n_threads > 0, "n_threads must not be 0.");
  check_arg(
      (score.rows() == static_cast<int64_t>(allowed_indices.size())) ||
          (allowed_indices.size() == 1u) || allowed_indices.empty(),
      "allowed_indices, if not empty, must have a size equal to X.rows()");
  std::vector<std::vector<score_and_index>> result(score.rows());
  std::vector<std::future<void>> workers;
  std::atomic<size_t> cursor(0);
  const size_t n_users = static_cast<size_t>(score.rows());
  for (size_t thread = 0; thread < std::min(n_threads, n_users); thread++) {
    workers.emplace_back(
        std::async([&score, cutoff, &allowed_indices, &cursor, &result]() {
          const int64_t n_rows = score.rows();
          const int64_t n_items = score.cols();
          std::vector<score_and_index> index_holder;
          index_holder.reserve(n_items);

          while (true) {
            int64_t current = cursor.fetch_add(1);
            if (current >= n_rows) {
              break;
            }

            std::vector<score_and_index> inserted;
            const Real *score_ptr = score.data() + n_items * current;

            index_holder.clear();
            if (!allowed_indices.empty()) {
              std::vector<int64_t>::const_iterator begin, end;
              if (allowed_indices.size() == 1u) {
                begin = allowed_indices[0].cbegin();
                end = allowed_indices[0].cend();
              } else {
                begin = allowed_indices.at(current).cbegin();
                end = allowed_indices.at(current).cend();
              }
              for (; begin != end; begin++) {
                auto item_index = *begin;
                if ((item_index < n_items) && (item_index >= 0)) {
                  index_holder.emplace_back(item_index, score_ptr[item_index]);
                }
              }
            } else {
              for (int64_t i = 0; i < n_items; i++) {
                index_holder.emplace_back(i, *(score_ptr++));
              }
            }
            std::partial_sort(
                index_holder.begin(),
                index_holder.begin() + std::min(cutoff, index_holder.size()),
                index_holder.end(), [](score_and_index i1, score_and_index i2) {
                  return i1.second > i2.second;
                });

            size_t items_recommended = 0;
            for (auto item_index : index_holder) {
              if (items_recommended >= cutoff) {
                break;
              }

              if (item_index.second == -std::numeric_limits<Real>::infinity()) {
                break;
              }
              result[current].emplace_back(item_index);
              items_recommended++;
            }
          }
        }));
  }
  workers.clear();
  return result;
}

} // namespace sparse_util
} // namespace irspack
