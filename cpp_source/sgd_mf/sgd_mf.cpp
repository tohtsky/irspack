#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <unordered_set>
#include <vector>

template <typename Real> struct SGDMF {
  using SparseMatrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
  using DenseMatrix =
      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using DenseVector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

  using Index = typename DenseMatrix::StorageIndex;
  using Sample = std::tuple<Index, Index, int64_t>;

  inline SGDMF(const SparseMatrix &X, int dim, int random_seed, Real lr,
               Real lambda, Real std, size_t n_negative)
      : X_(X), dim(dim), rng(random_seed), lr(lr), lambda(lambda),
        n_negative(n_negative) {
    X_.makeCompressed();
    P.resize(X_.rows(), dim);
    Q.resize(X_.cols(), dim);
    P_cache.resize(dim);
    Q_cache.resize(dim);
    P_b = DenseVector::Zero(X_.rows());
    Q_b = DenseVector::Zero(X_.cols());
    auto fill_normal = [this, std](DenseMatrix &U) {
      int rows = U.rows();
      int cols = U.cols();
      std::normal_distribution<Real> dist(0, std);
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          U(i, j) = dist(this->rng);
        }
      }
    };
    fill_normal(P);
    fill_normal(Q);

    size_t dsize = X_.nonZeros() + X_.rows() * n_negative;
    dataset.resize(dsize);
  }

  inline void start_epoch() {
    size_t cursor = 0;
    std::uniform_int_distribution<> dist(0, X_.cols() - 1);
    for (int u = 0; u < X_.rows(); u++) {
      for (typename SparseMatrix::InnerIterator iter(X_, u); iter; ++iter) {
        int j = iter.col();
        Sample q(u, j, 1);
        dataset[cursor++] = std::move(q);
      }
      for (size_t m_ = 0; m_ < n_negative; m_++) {
        dataset[cursor++] = {u, dist(rng), 0};
      }
    }
    if (static_cast<size_t>(X_.nonZeros() + X_.rows() * n_negative) != cursor) {
      throw std::runtime_error("somethong nasty");
    }
    std::shuffle(dataset.begin(), dataset.end(), rng);
  }

  inline Real run_epoch() {
    start_epoch();
    Real mean_loss = 0;
    for (auto &s : dataset) {
      mean_loss += sgd(s);
    }
    return mean_loss / dataset.size();
  }

  inline Real sgd(const Sample &s) {
    const Index &u = std::get<0>(s);
    const Index &i = std::get<1>(s);
    const int64_t &y = std::get<2>(s);
    P_cache.noalias() = P.row(u).transpose();
    Q_cache.noalias() = Q.row(i).transpose();
    Real score = (P_cache.transpose() * Q_cache) + bias + P_b(u) + Q_b(i);
    Real sigma_score;
    Real loss;
    if (score > 0) {
      sigma_score = 1 / (1 + std::exp(-score));
      loss = -std::log(sigma_score) + (1 - y) * score;
    } else {
      Real exp_score = std::exp(score);
      sigma_score = exp_score / (1 + exp_score);
      loss = -y * score + std::log(1 + exp_score);
    }

    Real grad = (y - sigma_score);

    P.row(u).noalias() += lr * (grad * Q_cache - lambda * P_cache).transpose();
    Q.row(i).noalias() += lr * (grad * P_cache - lambda * Q_cache).transpose();
    P_b(u) += lr * (grad - lambda * P_b(u));
    Q_b(i) += lr * (grad - lambda * Q_b(i));
    bias += lr * (grad - lambda * bias);
    return loss;
  }

  SparseMatrix X_;

  Real bias;
  DenseMatrix P, Q;
  DenseVector P_b, Q_b;
  DenseVector P_cache, Q_cache;
  int dim;

  std::vector<Sample> dataset;

private:
  std::mt19937 rng;

  Real lr, lambda;
  size_t n_negative;
};

namespace py = pybind11;
using std::vector;

PYBIND11_MODULE(_sgd_mf, m) {
  using Real = double;
  using MF = SGDMF<Real>;
  py::class_<MF>(m, "_MF")
      .def(py::init<const typename MF::SparseMatrix &, int, int, Real, Real,
                    Real, size_t>())
      .def("step", &MF::run_epoch)
      .def_readonly("dataset", &MF::dataset)
      .def_readwrite("P", &MF::P)
      .def_readwrite("Q", &MF::Q)
      .def_readwrite("P_b", &MF::P_b)
      .def_readwrite("Q_b", &MF::Q_b)
      .def_readwrite("bias", &MF::bias);
}
