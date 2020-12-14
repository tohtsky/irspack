#pragma once
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <atomic>
#include <cstddef>
#include <future>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace KNN {
/**
 * Given N \times n_features sparse matrix, computer
 * r x N
 * similarity matrix.
 */
template <typename Real, class SimilarityType> struct KNNComputer {

  using Triplet = typename Eigen::Triplet<Real>;
  using CSRMatrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
  using CSCMatrix = Eigen::SparseMatrix<Real, Eigen::ColMajor>;

  using CSRIterType = typename CSRMatrix::InnerIterator;
  using CSCIterType = typename CSCMatrix::InnerIterator;
  using DenseVector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
  KNNComputer(const CSRMatrix &X_arg, Real shrinkage, size_t n_thread)
      : X_t(X_arg.transpose()), n_thread(n_thread), N(X_arg.rows()),
        n_features(X_arg.cols()), shrinkage(shrinkage) {
    if (n_thread == 0) {
      std::invalid_argument("n_thread should be > 0");
    }
    X_t.makeCompressed();
  }

  CSRMatrix compute_similarity(const CSRMatrix &target, size_t top_k) const {
    if (target.cols() != this->n_features)
      throw std::invalid_argument("illegal # of feature.");
    CSRMatrix result(target.rows(), this->N);
    if (n_thread == 1) {

      auto triplets =
          this->compute_similarity_triple(target, 0, target.rows(), top_k);
      result.setFromTriplets(triplets.begin(), triplets.end());
      return result;
    } else {
      const size_t per_block_size = target.rows() / n_thread;
      const size_t remnant = target.rows() % n_thread;
      size_t job_start = 0;
      std::vector<std::future<std::vector<Triplet>>> thread_results;
      std::vector<Triplet> accumulated_result;
      for (size_t i = 0; i < n_thread; i++) {
        size_t block_size = per_block_size;
        if (i < remnant) {
          block_size += 1;
        }
        thread_results.push_back(
            std::async(std::launch::async,
                       [&target, job_start, block_size, this, top_k]() {
                         return this->compute_similarity_triple(
                             target, job_start, job_start + block_size, top_k);
                       }));
        job_start += block_size;
      }
      assert(job_start == target.rows());
      for (auto &thread_result : thread_results) {
        std::vector<Triplet> thread_result_ret = thread_result.get();
        accumulated_result.insert(accumulated_result.end(),
                                  thread_result_ret.begin(),
                                  thread_result_ret.end());
      }
      result.setFromTriplets(accumulated_result.begin(),
                             accumulated_result.end());
      return result;
    }
  }

  std::vector<Triplet> compute_similarity_triple(const CSRMatrix &target,
                                                 size_t start, size_t end,
                                                 size_t top_k) const {
    /*
     We want to compute n_item x n_item matrix
      S = X_t * X
     with each *column* restricted by top-k strategy.
     We devide the problem col-wise
     X_t * X[start, end]
    */
    CSRMatrix block_result_row = this->compute_sim_block(target, start, end);
    block_result_row.makeCompressed();

    using IndexType = typename CSCMatrix::StorageIndex;

    std::vector<Triplet> triples;
    std::vector<IndexType> buffer(this->N);

    const Real *data_ptr = block_result_row.valuePtr();
    auto index_ptr = block_result_row.innerIndexPtr();
    auto index_start_ptr = block_result_row.outerIndexPtr();

    for (IndexType row = 0; row < block_result_row.rows(); row++) {
      size_t nz_size = index_start_ptr[row + 1] - index_start_ptr[row];
      size_t col_size = std::min(nz_size, top_k);
      auto data_start = data_ptr + index_start_ptr[row];
      auto index_start = index_ptr + index_start_ptr[row];
      for (size_t i = 0; i < nz_size; i++) {
        buffer[i] = i;
      }
      std::sort(buffer.begin(), buffer.begin() + nz_size,
                [&data_start](IndexType &col1, IndexType &col2) {
                  return data_start[col1] > data_start[col2];
                });
      std::sort(buffer.begin(), buffer.begin() + col_size);
      for (size_t j = 0; j < col_size; j++) {
        triples.emplace_back(row + start,
                             static_cast<IndexType>(index_start[buffer[j]]),
                             data_start[buffer[j]]);
      }
    }
    return triples;
  }

public:
  CSCMatrix X_t;
  const size_t n_thread;
  const int N, n_features;

protected:
  Real shrinkage;
  DenseVector norms;

  inline CSRMatrix compute_sim_block(const CSRMatrix &target, size_t start,
                                     size_t end) const {
    if (start > end) {
      throw std::invalid_argument("start must be <= end");
    }

    return static_cast<const SimilarityType &>(*this).compute_similarity_imple(
        target, start, end);
  }

  /*
    CSRMatrix compute_sim_cosine(const CSRMatrix &target, size_t start,
                                 size_t end) const {
      int block_size = end - start;
      CSRMatrix result = target.middleRows(start, block_size) * (this->X_t);
      result.makeCompressed();
      for (int i = 0; i < block_size; i++) {
        Real norm = target.row(i).squaredNorm();
        for (typename CSRMatrix::InnerIterator iter(result, i); iter; ++iter) {
          iter.valueRef() /= (norms(iter.col()) * norm + this->shrinkage +
    1e-10);
        }
      }
    }

    CSRMatrix compute_sim_jaccard(const CSRMatrix &target, size_t start,
                                  size_t end) const {
      int block_size = end - start;
      CSRMatrix result = target.middleRows(start, block_size) * (this->X_t);
      result.makeCompressed();
      for (int i = 0; i < block_size; i++) {
        Real norm = target.row(i).sum();
        for (typename CSRMatrix::InnerIterator iter(result, i); iter; ++iter) {
          iter.valueRef() /= (norms(iter.col()) + norm - iter.valueRef() +
                              this->shrinkage + 1e-10);
        }
      }
    }

    CSRMatrix compute_sim_(const CSRMatrix &target, size_t start,
                           size_t end) const {
      int block_size = end - start;
      CSRMatrix result = target.middleRows(start, block_size) * (this->X_t);
      result.makeCompressed();
      for (int i = 0; i < block_size; i++) {
        Real norm = target.row(i).sum();
        for (typename CSRMatrix::InnerIterator iter(result, i); iter; ++iter) {
          iter.valueRef() /= (norms(iter.col()) + norm - iter.valueRef() +
                              this->shrinkage + 1e-10);
        }
      }
    }
    */
};

} // namespace KNN