#pragma once
#include "knn.hpp"
#include <cstdlib>
#include <stdexcept>

namespace KNN {
template <typename Real>
struct CosineSimilarityComputer
    : KNNComputer<Real, CosineSimilarityComputer<Real>> {
  using Base = KNNComputer<Real, CosineSimilarityComputer>;
  using CSCMatrix = typename Base::CSCMatrix;
  using CSRMatrix = typename Base::CSRMatrix;

protected:
  bool normalize;

public:
  inline CosineSimilarityComputer(const CSRMatrix &X_arg, Real shrinkage,
                                  bool normalize, size_t n_thread)
      : Base(X_arg, shrinkage, n_thread), normalize(normalize) {
    for (int i = 0; i < this->N; i++) {
      this->norms(i) = this->X_t.col(i).norm();
    }
  }
  inline CSRMatrix compute_similarity_imple(const CSRMatrix &target,
                                            size_t start, size_t end) const {
    // target: something x U, X_t: U x I
    int block_size = end - start;
    CSRMatrix result = target.middleRows(start, block_size) * (this->X_t);
    result.makeCompressed();
    if (!this->normalize) {
      return result;
    }
    for (int i = 0; i < block_size; i++) {
      Real target_norm = target.row(start + i).norm();
      for (typename CSRMatrix::InnerIterator iter(result, i); iter; ++iter) {
        iter.valueRef() /=
            (this->norms(iter.col()) * target_norm + this->shrinkage + 1e-6);
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
      this->norms(i) = this->X_t.col(i).sum();
    }
  }

  inline CSRMatrix compute_similarity_imple(const CSRMatrix &target,
                                            size_t start, size_t end) const {
    int block_size = end - start;
    CSRMatrix target_bin(target.middleRows(start, block_size));
    for (int i = 0; i < target_bin.rows(); i++) {
      for (typename CSRMatrix::InnerIterator iter(target_bin, i); iter;
           ++iter) {
        iter.valueRef() = 1;
      }
    }

    CSRMatrix result = target_bin * (this->X_t);
    result.makeCompressed();
    for (int i = 0; i < block_size; i++) {
      Real target_norm = target_bin.row(i).sum();
      for (typename CSRMatrix::InnerIterator iter(result, i); iter; ++iter) {
        iter.valueRef() /= (this->norms(iter.col()) + target_norm -
                            iter.valueRef() + this->shrinkage + 1e-6);
      }
    }
    return result;
  }
};

template <typename Real>
struct AsymmetricCosineSimilarityComputer
    : KNNComputer<Real, AsymmetricCosineSimilarityComputer<Real>> {
  using Base = KNNComputer<Real, AsymmetricCosineSimilarityComputer>;
  using CSCMatrix = typename Base::CSCMatrix;
  using CSRMatrix = typename Base::CSRMatrix;
  using DenseVector = typename Base::DenseVector;

protected:
  Real alpha;

public:
  inline AsymmetricCosineSimilarityComputer(const CSRMatrix &X_arg,
                                            Real shrinkage, Real alpha,
                                            size_t n_thread)
      : Base(X_arg, shrinkage, n_thread), alpha(alpha) {
    if ((alpha > 1) || (alpha < 0)) {
      throw std::invalid_argument("alpha must be in [0, 1]");
    }
    for (int i = 0; i < this->N; i++) {
      Real norm = this->X_t.col(i).squaredNorm();
      this->norms(i) = norm;
    }
    this->norms.array() = this->norms.array().pow((1 - alpha));
  }
  inline CSRMatrix compute_similarity_imple(const CSRMatrix &target,
                                            size_t start, size_t end) const {
    int block_size = end - start;
    CSRMatrix result = target.middleRows(start, block_size) * (this->X_t);
    result.makeCompressed();
    for (int i = 0; i < block_size; i++) {
      Real target_norm =
          std::pow(target.row(start + i).squaredNorm(), this->alpha);
      for (typename CSRMatrix::InnerIterator iter(result, i); iter; ++iter) {
        iter.valueRef() /=
            (this->norms(iter.col()) * target_norm + this->shrinkage + 1e-6);
      }
    }
    return result;
  }
};

template <typename Real>
struct P3alphaComputer : KNNComputer<Real, P3alphaComputer<Real>> {
  using Base = KNNComputer<Real, P3alphaComputer>;
  using CSCMatrix = typename Base::CSCMatrix;
  using CSRMatrix = typename Base::CSRMatrix;
  using DenseVector = typename Base::DenseVector;

protected:
  Real alpha;

public:
  inline P3alphaComputer(const CSRMatrix &X_arg, Real alpha, bool normalize,
                         size_t n_thread)
      : Base(X_arg, 0, n_thread), alpha(alpha) {
    // We want ItU * UtI
    // and each rows to be normalized & each *columns* to be top-K constrained,
    // so the computation should be
    // (ItU * UtI) [:, start:end]
    // = (UtI[start:end, : ] ^T ItU ^T  )^T
    // So the argument should be ItU (rows sum to 1), and X_t should be ItU ^ T
    // (cols sum to 1)

    // this->X_t
    DenseVector norm_temp(this->X_t.cols()); // n_users
    norm_temp.array() = 0;

    for (int i = 0; i < this->X_t.cols(); i++) {
      for (typename CSCMatrix::InnerIterator iter(this->X_t, i); iter; ++iter) {
        iter.valueRef() = std::pow(iter.valueRef(), this->alpha);
        norm_temp(iter.col()) += iter.value();
      }
    }
    for (int i = 0; i < this->X_t.cols(); i++) {
      for (typename CSCMatrix::InnerIterator iter(this->X_t, i); iter; ++iter) {
        iter.valueRef() /= norm_temp(iter.col());
      }
    }
  }

  inline CSCMatrix compute_W(const CSRMatrix &arg, size_t top_k) const {
    // normalize along column
    DenseVector norm_temp(arg.cols()); // n_users
    norm_temp.array() = 0;
    CSRMatrix target(arg);

    for (int i = 0; i < target.rows(); i++) {
      for (typename CSRMatrix::InnerIterator iter(target, i); iter; ++iter) {
        iter.valueRef() = std::pow(iter.valueRef(), this->alpha);
        norm_temp(iter.col()) += iter.value();
      }
    }
    for (int i = 0; i < target.rows(); i++) {
      for (typename CSRMatrix::InnerIterator iter(target, i); iter; ++iter) {
        iter.valueRef() /= norm_temp(iter.col());
      }
    }
    return this->compute_similarity(target, top_k).transpose();
  }

  inline CSRMatrix compute_similarity_imple(const CSRMatrix &target,
                                            size_t start, size_t end) const {
    // we will be given X_{IU}
    int block_size = end - start;
    CSRMatrix result = target.middleRows(start, block_size) * (this->X_t);
    result.makeCompressed();
    return result;
  }
};

} // namespace KNN
