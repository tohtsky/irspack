#pragma once
#include <cstddef>
#include <cstdlib>
#include <stdexcept>

#include "../argcheck.hpp"
#include "knn.hpp"
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
                                  bool normalize, size_t n_threads,
                                  size_t max_chunk_size)
      : Base(X_arg, shrinkage, n_threads, max_chunk_size),
        normalize(normalize) {
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
                                            size_t n_threads,
                                            size_t max_chunk_size)
      : Base(X_arg, shrinkage, n_threads, max_chunk_size), alpha(alpha) {
    irspack::check_arg_doubly_bounded<Real>(alpha, 0, 1, "alpha");
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
struct JaccardSimilarityComputer
    : KNNComputer<Real, JaccardSimilarityComputer<Real>> {
  using Base = KNNComputer<Real, JaccardSimilarityComputer>;
  using CSCMatrix = typename Base::CSCMatrix;
  using CSRMatrix = typename Base::CSRMatrix;
  inline JaccardSimilarityComputer(const CSRMatrix &X_arg, Real shrinkage,
                                   size_t n_threads, size_t max_chunk_size)
      : Base(X_arg, shrinkage, n_threads, max_chunk_size) {
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
struct TverskyIndexComputer : KNNComputer<Real, TverskyIndexComputer<Real>> {
  using Base = KNNComputer<Real, TverskyIndexComputer<Real>>;
  using CSCMatrix = typename Base::CSCMatrix;
  using CSRMatrix = typename Base::CSRMatrix;

protected:
  Real alpha, beta;

public:
  inline TverskyIndexComputer(const CSRMatrix &X_arg, Real shrinkage,
                              Real alpha, Real beta, size_t n_threads,
                              size_t max_chunk_size)
      : Base(X_arg, shrinkage, n_threads, max_chunk_size), alpha(alpha),
        beta(beta) {
    irspack::check_arg_lower_bounded<Real>(alpha, 0, "alpha");
    irspack::check_arg_lower_bounded<Real>(beta, 0, "beta");

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
        Real itv = iter.value();
        iter.valueRef() /=
            (itv + this->beta * (this->norms(iter.col()) - itv) +
             this->alpha * (target_norm - itv) + this->shrinkage + 1e-6);
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
  inline P3alphaComputer(const CSRMatrix &X_arg, Real alpha, size_t n_threads,
                         size_t max_chunk_size)
      : Base(X_arg, 0, n_threads, max_chunk_size), alpha(alpha) {
    irspack::check_arg_lower_bounded<Real>(alpha, 0, "alpha");
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

template <typename Real>
struct RP3betaComputer : KNNComputer<Real, RP3betaComputer<Real>> {
  using Base = KNNComputer<Real, RP3betaComputer>;
  using CSCMatrix = typename Base::CSCMatrix;
  using CSRMatrix = typename Base::CSRMatrix;
  using DenseVector = typename Base::DenseVector;

protected:
  Real alpha, beta;

public:
  inline RP3betaComputer(const CSRMatrix &X_arg, Real alpha, Real beta,
                         size_t n_threads, size_t max_chunk_size)
      : Base(X_arg, 0, n_threads, max_chunk_size), alpha(alpha), beta(beta) {
    irspack::check_arg_lower_bounded<Real>(alpha, 0, "alpha");
    irspack::check_arg_lower_bounded<Real>(beta, 0, "beta");
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
        norm_temp(iter.col()) += iter.valueRef();
      }
    }
    for (int i = 0; i < this->X_t.cols(); i++) {
      for (typename CSCMatrix::InnerIterator iter(this->X_t, i); iter; ++iter) {
        iter.valueRef() /= norm_temp(iter.col());
      }
    }
  }

  inline CSCMatrix compute_W(const CSRMatrix &arg, size_t top_k) const {
    // arg has n_item x n_user shape

    CSRMatrix target(arg);
    target.makeCompressed();

    DenseVector popularity_temp(arg.rows()); // n_items
    popularity_temp.array() = 0;
    for (int i = 0; i < target.rows(); i++) {
      for (typename CSRMatrix::InnerIterator iter(target, i); iter; ++iter) {
        popularity_temp(iter.row()) += iter.value();
      }
    }
    popularity_temp.array() = popularity_temp.array().pow(this->beta);
    // normalize along column
    DenseVector norm_temp(arg.cols()); // n_users
    norm_temp.array() = 0;

    for (int i = 0; i < target.rows(); i++) {
      for (typename CSRMatrix::InnerIterator iter(target, i); iter; ++iter) {
        iter.valueRef() = std::pow(iter.valueRef(), this->alpha);
        norm_temp(iter.col()) += iter.valueRef();
      }
    }
    for (int i = 0; i < target.rows(); i++) {
      for (typename CSRMatrix::InnerIterator iter(target, i); iter; ++iter) {
        iter.valueRef() /=
            (norm_temp(iter.col()) * popularity_temp(iter.row()));
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
