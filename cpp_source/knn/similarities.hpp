#pragma once
#include "knn.hpp"
#include <cstdlib>

namespace KNN {
template <typename Real>
struct CosineSimilarityComputer
    : KNNComputer<Real, CosineSimilarityComputer<Real>> {
  using Base = KNNComputer<Real, CosineSimilarityComputer>;
  using CSCMatrix = typename Base::CSCMatrix;
  using CSRMatrix = typename Base::CSRMatrix;
  inline CosineSimilarityComputer(const CSRMatrix &X_arg, Real shrinkage,
                                  size_t n_thread)
      : Base(X_arg, shrinkage, n_thread) {
    for (int i = 0; i < this->N; i++) {
      this->norms(i) = std::sqrt(this->X_t.col(i).squaredNorm());
    }
  }
  inline CSRMatrix compute_similarity_imple(const CSRMatrix &target,
                                            size_t start, size_t end) const {
    int block_size = end - start;
    CSRMatrix result = target.middleRows(start, block_size) * (this->X_t);
    result.makeCompressed();
    for (int i = 0; i < block_size; i++) {
      Real norm = this->norms(i);
      for (typename CSRMatrix::InnerIterator iter(result, i); iter; ++iter) {
        iter.valueRef() /=
            (this->norms(iter.col()) * norm + this->shrinkage + 1e-10);
      }
    }
    return result;
  }
};

template <typename Real>
struct JaccardSimilarityComputer
    : KNNComputer<Real, JaccardSimilarityComputer<Real>> {
  using Base = KNNComputer<Real, JaccardSimilarityComputer>;
  using CSCMatrix = typename Base::CSCMatrix;
  using CSRMatrix = typename Base::CSRMatrix;
  inline JaccardSimilarityComputer(const CSRMatrix &X_arg, Real shrinkage,
                                   size_t n_thread)
      : Base(X_arg, shrinkage, n_thread) {
    for (int i = 0; i < this->X_t.cols(); i++) {
      for (typename CSCMatrix::InnerIterator iter(this->X_t, i); iter; ++iter) {
        iter.valueRef() = 1;
      }
    }
    for (int i = 0; i < this->N; i++) {
      this->norms(i) = std::sqrt(this->X_t.col(i).sum());
    }
  }
  inline CSRMatrix compute_similarity_imple(const CSRMatrix &target,
                                            size_t start, size_t end) const {
    int block_size = end - start;
    CSRMatrix result = target.middleRows(start, block_size) * (this->X_t);
    result.makeCompressed();
    for (int i = 0; i < block_size; i++) {
      Real norm = this->norms(i);
      for (typename CSRMatrix::InnerIterator iter(result, i); iter; ++iter) {
        iter.valueRef() /= (this->norms(iter.col()) + norm - iter.valueRef() +
                            this->shrinkage + 1e-10);
      }
    }
    return result;
  }
};
} // namespace KNN