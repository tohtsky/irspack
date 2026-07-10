#pragma once
#include "../argcheck.hpp"
#include "IALSLearningConfig.hpp"
#include "definitions.hpp"
#include <Eigen/Cholesky>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iostream>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <variant>
#include <vector>

namespace irspack {
namespace ials {
using namespace std;
// using SymmetricMatrix = Eigen::SelfAdjointView<DenseMatrix &, Eigen::Upper>;
using SymmetricMatrix =
    decltype(DenseMatrix{0, 0}.selfadjointView<Eigen::Upper>());
using FeatureMatrix = std::variant<SparseMatrix, DenseMatrix>;
template <size_t n_batch_max = 64> struct BatchedRankUpdater {
  inline BatchedRankUpdater(size_t dim)
      : n_batch(0u), buffer(n_batch_max, dim) {}
  inline void add_row(SymmetricMatrix &target, DenseVector &row, Real c) {
    buffer.row(n_batch++).noalias() = std::sqrt(c) * row.transpose();
    if (n_batch >= n_batch_max) {
      consume(target);
    }
  }
  inline void consume(SymmetricMatrix &target) {
    if (n_batch > 0u) {
      target.rankUpdate(buffer.middleRows(0, n_batch).adjoint(),
                        static_cast<Real>(1.0));
      this->clear();
    }
  }
  inline void clear() { n_batch = 0; }

private:
  size_t n_batch;
  DenseMatrix buffer;
};

struct Solver {
  Solver(const IALSModelConfig &config)
      : P(config.K, config.K), p_initialized(false) {}

  inline void initialize(DenseMatrix &factor, const IALSModelConfig &config) {

    if (config.init_stdev > 0) {
      std::mt19937 gen(config.random_seed);
      std::normal_distribution<Real> dist(0.0, config.init_stdev /
                                                   std::sqrt(factor.cols()));
      for (int i = 0; i < factor.rows(); i++) {
        for (int k = 0; k < factor.cols(); k++) {
          factor(i, k) = dist(gen);
        }
      }
    }
  }

  inline void prepare_p(const DenseMatrix &other_factor,
                        const IALSModelConfig &model_config,
                        const SolverConfig &solver_config) {
    const int64_t mb_size = 16;
    P = DenseMatrix::Zero(other_factor.cols(), other_factor.cols());

    std::atomic<int64_t> cursor{static_cast<size_t>(0)};

    std::vector<std::future<DenseMatrix>> workers;
    for (size_t i = 0; i < solver_config.n_threads; i++) {
      workers.emplace_back(std::async(std::launch::async, [&other_factor,
                                                           &cursor, mb_size]() {
        DenseMatrix P_local =
            DenseMatrix::Zero(other_factor.cols(), other_factor.cols());
        while (true) {
          int64_t cursor_local = cursor.fetch_add(mb_size);
          if (cursor_local >= other_factor.rows())
            break;
          int64_t block_end =
              std::min<int64_t>(cursor_local + mb_size,
                                static_cast<int64_t>(other_factor.rows()));
          P_local.noalias() +=
              other_factor.middleRows(cursor_local, block_end - cursor_local)
                  .transpose() *
              other_factor.middleRows(cursor_local, block_end - cursor_local);
        }
        return P_local;
      }));
    }
    for (auto &w : workers) {
      P.noalias() += w.get();
    }
    P *= model_config.alpha0;
    p_initialized = true;
  }

  inline Real compute_reg(const int64_t nnz, const int64_t other_size,
                          const IALSModelConfig &config) const {
    return config.reg * std::pow(config.alpha0 * other_size + nnz, config.nu);
  }

  inline DenseMatrix X_to_vector(const SparseMatrix &X,
                                 const DenseMatrix &other_factor,
                                 const IALSModelConfig &config,
                                 const SolverConfig &solver_config) const {
    if (X.cols() != other_factor.rows()) {
      std::stringstream ss;
      ss << "Shape mismatch: X.cols() = " << X.cols()
         << " but other.factor.rows() = " << other_factor.rows() << ".";
      throw std::invalid_argument(ss.str());
    }
    DenseMatrix result = DenseMatrix::Zero(X.rows(), P.rows());
    if (X.isCompressed()) {
      this->step(result, X, other_factor, config, solver_config);
    } else {
      SparseMatrix X_compressed = X;
      X_compressed.makeCompressed();
      this->step(result, X_compressed, other_factor, config, solver_config);
    }
    return result;
  }

  inline DenseMatrix X_to_vector_with_prior(
      const SparseMatrix &X, const DenseMatrix &other_factor,
      const DenseMatrix &prior, const IALSModelConfig &config,
      const SolverConfig &solver_config) const {
    if (X.cols() != other_factor.rows()) {
      std::stringstream ss;
      ss << "Shape mismatch: X.cols() = " << X.cols()
         << " but other.factor.rows() = " << other_factor.rows() << ".";
      throw std::invalid_argument(ss.str());
    }
    if (prior.rows() != X.rows() || prior.cols() != P.rows()) {
      throw std::invalid_argument("Feature prior shape does not match X.");
    }
    DenseMatrix result = prior;
    if (X.isCompressed()) {
      this->step_with_prior(result, X, other_factor, prior, config,
                            solver_config);
    } else {
      SparseMatrix X_compressed = X;
      X_compressed.makeCompressed();
      this->step_with_prior(result, X_compressed, other_factor, prior, config,
                            solver_config);
    }
    return result;
  }

private:
  inline void step_cg(DenseMatrix &target_factor, const SparseMatrix &X,
                      const DenseMatrix &other_factor,
                      const IALSModelConfig &config,
                      const SolverConfig &solver_config) const {

    std::vector<std::thread> workers;

    std::atomic<int> cursor(0);
    for (size_t ind = 0; ind < solver_config.n_threads; ind++) {
      workers.emplace_back([this, &target_factor, &cursor, &X, &other_factor,
                            &config, &solver_config]() {
        DenseVector b(P.rows()), x(P.rows()), r(P.rows()), p(P.rows()),
            Ap(P.rows());
        Real observation_bias =
            config.loss_type == LossType::IALSPP ? 0.0 : config.alpha0;

        while (true) {
          int cursor_local = cursor.fetch_add(1);
          if (cursor_local >= target_factor.rows()) {
            break;
          }

          b.array() = 0;

          x = target_factor.row(cursor_local)
                  .transpose(); // use the previous value

          size_t nnz = 0;
          for (SparseMatrix::InnerIterator it(X, cursor_local); it; ++it) {
            b.noalias() += (observation_bias + it.value()) *
                           other_factor.row(it.col()).transpose();
            nnz++;
          }

          const Real regularization_this =
              this->compute_reg(nnz, other_factor.rows(), config);
          if (nnz == 0u) {
            target_factor.row(cursor_local).array() = 0;
            continue;
          }
          r = b - P * x;
          r -= regularization_this * x;
          for (SparseMatrix::InnerIterator it(X, cursor_local); it; ++it) {
            Real vdotx = other_factor.row(it.col()) * x;
            r.noalias() -=
                it.value() * vdotx * other_factor.row(it.col()).transpose();
          }

          p = r;

          size_t cg_max_iter = solver_config.max_cg_steps == 0u
                                   ? P.rows()
                                   : solver_config.max_cg_steps;

          for (size_t cg_iter = 0; cg_iter < cg_max_iter; cg_iter++) {
            Real r2 = r.squaredNorm();
            Ap = P * p;
            Ap += regularization_this * p;
            for (SparseMatrix::InnerIterator it(X, cursor_local); it; ++it) {
              Real vdotp = other_factor.row(it.col()) * p;
              Ap +=
                  (it.value() * vdotp) * other_factor.row(it.col()).transpose();
            }

            Real alpha_denom = p.transpose() * Ap;
            Real alpha = r2 / alpha_denom;
            x.noalias() += alpha * p;
            r.noalias() -= alpha * Ap;
            if (r.norm() <= 1e-10) {
              break;
            }
            Real beta = r.squaredNorm() / r2;
            p.noalias() = r + beta * p;
          }
          target_factor.row(cursor_local) = x.transpose();
        }
      });
    }
    for (auto &w : workers) {
      w.join();
    }
  }

  inline void step_cholesky(DenseMatrix &target_factor, const SparseMatrix &X,
                            const DenseMatrix &other_factor,
                            const IALSModelConfig &config,
                            const SolverConfig &solver_config) const {

    std::vector<std::thread> workers;

    std::atomic<int> cursor(0);
    for (size_t ind = 0; ind < solver_config.n_threads; ind++) {
      workers.emplace_back([this, &target_factor, &cursor, &X, &other_factor,
                            &config]() {
        BatchedRankUpdater<> bupdater(P.rows());
        DenseMatrix P_local(P.rows(), P.cols());
        DenseVector B(P.rows());
        DenseVector vcache(P.rows()), r(P.rows());
        Real observation_bias =
            config.loss_type == LossType::IALSPP ? 0.0 : config.alpha0;
        while (true) {
          int cursor_local = cursor.fetch_add(1);
          if (cursor_local >= target_factor.rows()) {
            break;
          }
          P_local.noalias() = P;
          SymmetricMatrix P_adjoint = P_local.selfadjointView<Eigen::Upper>();

          B.array() = static_cast<Real>(0);
          int64_t nnz = 0;
          for (SparseMatrix::InnerIterator it(X, cursor_local); it; ++it) {
            int64_t other_index = it.col();
            vcache = other_factor.row(other_index).transpose();
            bupdater.add_row(P_adjoint, vcache, it.value());
            B.noalias() += (observation_bias + it.value()) * vcache;
            nnz++;
          }
          bupdater.consume(P_adjoint);
          const Real regularization_this =
              this->compute_reg(nnz, other_factor.rows(), config);

          for (int64_t i = 0; i < P.rows(); i++) {
            P_local(i, i) += regularization_this;
          }

          Eigen::LLT<Eigen::Ref<DenseMatrix>, Eigen::Upper> llt(P_local);
          target_factor.row(cursor_local) = llt.solve(B);
        }
      });
    }
    for (auto &w : workers) {
      w.join();
    }
  }

  inline void step_cg_with_prior(
      DenseMatrix &target_factor, const SparseMatrix &X,
      const DenseMatrix &other_factor, const DenseMatrix &prior,
      const IALSModelConfig &config,
      const SolverConfig &solver_config) const {
    if (prior.rows() != target_factor.rows() ||
        prior.cols() != target_factor.cols()) {
      throw std::invalid_argument("Feature prior shape does not match factor.");
    }
    std::vector<std::thread> workers;
    std::atomic<int> cursor(0);
    for (size_t ind = 0; ind < solver_config.n_threads; ind++) {
      workers.emplace_back([this, &target_factor, &cursor, &X, &other_factor,
                            &prior, &config, &solver_config]() {
        DenseVector b(P.rows()), x(P.rows()), r(P.rows()), p(P.rows()),
            Ap(P.rows());
        const Real observation_bias =
            config.loss_type == LossType::IALSPP ? 0.0 : config.alpha0;
        while (true) {
          const int row = cursor.fetch_add(1);
          if (row >= target_factor.rows())
            break;

          int64_t nnz = X.outerIndexPtr()[row + 1] - X.outerIndexPtr()[row];
          const Real diagonal = compute_reg(nnz, other_factor.rows(), config);
          b.noalias() = diagonal * prior.row(row).transpose();
          x = target_factor.row(row).transpose();
          for (SparseMatrix::InnerIterator it(X, row); it; ++it) {
            b.noalias() += (observation_bias + it.value()) *
                           other_factor.row(it.col()).transpose();
          }
          r = b - P * x - diagonal * x;
          for (SparseMatrix::InnerIterator it(X, row); it; ++it) {
            const Real vdotx = other_factor.row(it.col()) * x;
            r.noalias() -=
                it.value() * vdotx * other_factor.row(it.col()).transpose();
          }
          p = r;
          const size_t cg_max_iter = solver_config.max_cg_steps == 0u
                                         ? P.rows()
                                         : solver_config.max_cg_steps;
          for (size_t cg_iter = 0; cg_iter < cg_max_iter; ++cg_iter) {
            const Real r2 = r.squaredNorm();
            if (r2 <= static_cast<Real>(1e-20))
              break;
            Ap = P * p + diagonal * p;
            for (SparseMatrix::InnerIterator it(X, row); it; ++it) {
              const Real vdotp = other_factor.row(it.col()) * p;
              Ap.noalias() += it.value() * vdotp *
                              other_factor.row(it.col()).transpose();
            }
            const Real alpha = r2 / p.dot(Ap);
            x.noalias() += alpha * p;
            r.noalias() -= alpha * Ap;
            const Real next_r2 = r.squaredNorm();
            if (next_r2 <= static_cast<Real>(1e-20))
              break;
            p.noalias() = r + (next_r2 / r2) * p;
          }
          target_factor.row(row) = x.transpose();
        }
      });
    }
    for (auto &worker : workers)
      worker.join();
  }

  inline void step_cholesky_with_prior(
      DenseMatrix &target_factor, const SparseMatrix &X,
      const DenseMatrix &other_factor, const DenseMatrix &prior,
      const IALSModelConfig &config,
      const SolverConfig &solver_config) const {
    if (prior.rows() != target_factor.rows() ||
        prior.cols() != target_factor.cols()) {
      throw std::invalid_argument("Feature prior shape does not match factor.");
    }
    std::vector<std::thread> workers;
    std::atomic<int> cursor(0);
    for (size_t ind = 0; ind < solver_config.n_threads; ind++) {
      workers.emplace_back([this, &target_factor, &cursor, &X, &other_factor,
                            &prior, &config]() {
        BatchedRankUpdater<> bupdater(P.rows());
        DenseMatrix P_local(P.rows(), P.cols());
        DenseVector B(P.rows()), vcache(P.rows());
        const Real observation_bias =
            config.loss_type == LossType::IALSPP ? 0.0 : config.alpha0;
        while (true) {
          const int row = cursor.fetch_add(1);
          if (row >= target_factor.rows())
            break;
          P_local.noalias() = P;
          SymmetricMatrix P_adjoint = P_local.selfadjointView<Eigen::Upper>();
          const int64_t nnz =
              X.outerIndexPtr()[row + 1] - X.outerIndexPtr()[row];
          const Real diagonal = compute_reg(nnz, other_factor.rows(), config);
          B.noalias() = diagonal * prior.row(row).transpose();
          for (SparseMatrix::InnerIterator it(X, row); it; ++it) {
            vcache = other_factor.row(it.col()).transpose();
            bupdater.add_row(P_adjoint, vcache, it.value());
            B.noalias() += (observation_bias + it.value()) * vcache;
          }
          bupdater.consume(P_adjoint);
          P_local.diagonal().array() += diagonal;
          Eigen::LLT<Eigen::Ref<DenseMatrix>, Eigen::Upper> llt(P_local);
          if (llt.info() != Eigen::Success)
            throw std::runtime_error("Cholesky decomposition failed.");
          target_factor.row(row) = llt.solve(B);
        }
      });
    }
    for (auto &worker : workers)
      worker.join();
  }

  inline DenseVector _prediction(const SparseMatrix &X_compressed,
                                 const DenseMatrix &target_factor,
                                 const DenseMatrix &other_factor,
                                 const SolverConfig &solver_config) const {
    size_t NNZ = X_compressed.nonZeros();
    DenseVector predictions(NNZ);
    std::vector<std::thread> workers;
    std::atomic<int> cursor(0);
    for (size_t ind = 0; ind < solver_config.n_threads; ind++) {
      workers.emplace_back([this, &target_factor, &cursor, &X_compressed,
                            &other_factor, &solver_config, &predictions]() {
        while (true) {
          int cursor_local = cursor.fetch_add(1);
          if (cursor_local >= target_factor.rows()) {
            break;
          }
          auto start = X_compressed.outerIndexPtr()[cursor_local];
          auto end = X_compressed.outerIndexPtr()[cursor_local + 1];
          auto inner_index_ptr = X_compressed.innerIndexPtr() + start;
          for (int inner_cursor = start; inner_cursor < end; inner_cursor++) {
            predictions(inner_cursor) =
                target_factor.row(cursor_local)
                    .dot(other_factor.row(*inner_index_ptr));
            inner_index_ptr++;
          }
        }
      });
    }

    for (auto &worker : workers) {
      worker.join();
    }

    return predictions;
  }

  inline void _step_dimrange(const int dim_start, const int dim_end,
                             DenseVector &predictions,
                             DenseMatrix &target_factor,
                             const SparseMatrix &X_compressed,
                             const DenseMatrix &other_factor,
                             const IALSModelConfig &config,
                             const SolverConfig &solver_config) const {

    int subspace_dim = dim_end - dim_start;
    std::vector<std::thread> workers;
    std::atomic<int> cursor(0);
    DenseMatrix target_factor_subspaced =
        target_factor.middleCols(dim_start, subspace_dim);
    const DenseMatrix other_factor_subspaced =
        other_factor.middleCols(dim_start, subspace_dim);

    const DenseMatrix P_quadratic =
        P.block(dim_start, dim_start, subspace_dim, subspace_dim);
    const DenseMatrix P_subspaced = P.middleRows(dim_start, subspace_dim);

    for (size_t ind = 0; ind < solver_config.n_threads; ind++) {
      workers.emplace_back([this, subspace_dim, dim_start, dim_end, P_quadratic,
                            P_subspaced, &target_factor_subspaced,
                            &target_factor, &other_factor_subspaced, &cursor,
                            &X_compressed, &config, solver_config,
                            &predictions]() {
        DenseMatrix P_local(subspace_dim, subspace_dim);
        DenseVector vcache(subspace_dim);
        DenseVector B(subspace_dim);
        BatchedRankUpdater<> bupdater(subspace_dim);
        Real observation_bias =
            config.loss_type == LossType::IALSPP ? 0.0 : config.alpha0;

        while (true) {
          int cursor_local = cursor.fetch_add(1);
          if (cursor_local >= target_factor_subspaced.rows()) {
            break;
          }

          P_local = P_quadratic;
          SymmetricMatrix P_adjoint = P_local.selfadjointView<Eigen::Upper>();

          const auto inner_cursor_start =
              X_compressed.outerIndexPtr()[cursor_local];
          auto inner_cursor_end =
              X_compressed.outerIndexPtr()[cursor_local + 1];
          auto nnz = inner_cursor_end - inner_cursor_start;
          auto reg =
              this->compute_reg(nnz, other_factor_subspaced.rows(), config);

          B.noalias() =
              P_subspaced * target_factor.row(cursor_local).transpose();

          B.noalias() +=
              reg * target_factor_subspaced.row(cursor_local).transpose();

          int64_t inner_cursor = inner_cursor_start;

          for (SparseMatrix::InnerIterator it(X_compressed, cursor_local); it;
               ++it) {
            int64_t other_index = it.col();
            vcache = other_factor_subspaced.row(other_index).transpose();
            Real residual = (it.value() * (predictions(inner_cursor++) - 1) -
                             observation_bias);
            bupdater.add_row(P_adjoint, vcache, it.value());
            B.noalias() += residual * vcache;
          }
          bupdater.consume(P_adjoint);
          for (int64_t i = 0; i < P_local.rows(); i++) {
            P_local(i, i) += reg;
          }

          Eigen::LLT<Eigen::Ref<DenseMatrix>, Eigen::Upper> llt(P_local);
          // difference. Current - vcache = newvector
          vcache = llt.solve(B);
          target_factor_subspaced.row(cursor_local) -= vcache;

          inner_cursor = inner_cursor_start;
          for (SparseMatrix::InnerIterator it(X_compressed, cursor_local); it;
               ++it) {
            int64_t other_index = it.col();
            predictions.coeffRef(inner_cursor++) -=
                vcache.dot(other_factor_subspaced.row(other_index).transpose());
          }
        }
      });
    }
    for (auto &worker : workers) {
      worker.join();
    }
    target_factor.middleCols(dim_start, subspace_dim) = target_factor_subspaced;
  }

  inline void step_ialspp(DenseMatrix &target_factor,
                          const SparseMatrix &X_compressed,
                          const DenseMatrix &other_factor,
                          const IALSModelConfig &config,
                          const SolverConfig &solver_config) const {
    for (size_t iter = 0; iter < solver_config.ialspp_iteration; iter++) {

      auto predictions = this->_prediction(X_compressed, target_factor,
                                           other_factor, solver_config);

      const size_t dim_all = target_factor.cols();
      for (size_t cursor = 0; cursor < dim_all;
           cursor += solver_config.ialspp_subspace_dimension) {
        size_t cursor_end =
            std::min(cursor + solver_config.ialspp_subspace_dimension, dim_all);
        this->_step_dimrange(cursor, cursor_end, predictions, target_factor,
                             X_compressed, other_factor, config, solver_config);
      }
    }
  }

  inline void step_icd(DenseMatrix &target_factor,
                       const SparseMatrix &X_compressed,
                       const DenseMatrix &other_factor,
                       const IALSModelConfig &config,
                       const SolverConfig &solver_config) const {
    for (size_t iter = 0; iter < solver_config.ialspp_iteration; iter++) {

      auto predictions = this->_prediction(X_compressed, target_factor,
                                           other_factor, solver_config);

      const size_t dim_all = target_factor.cols();
      for (size_t cursor = 0; cursor < dim_all; cursor++) {
        this->_step_icd(cursor, predictions, target_factor, X_compressed,
                        other_factor, config, solver_config);
      }
    }
  }
  inline void _step_icd(const int dim_start, DenseVector &predictions,
                        DenseMatrix &target_factor,
                        const SparseMatrix &X_compressed,
                        const DenseMatrix &other_factor,
                        const IALSModelConfig &config,
                        const SolverConfig &solver_config) const {

    std::vector<std::thread> workers;
    std::atomic<int> cursor(0);
    DenseVector target_factor_subspaced = target_factor.col(dim_start);
    const DenseVector other_factor_subspaced = other_factor.col(dim_start);

    const Real P_quadratic = P(dim_start, dim_start);
    const DenseVector P_subspaced = P.row(dim_start);

    for (size_t ind = 0; ind < solver_config.n_threads; ind++) {
      workers.emplace_back([this, dim_start, P_quadratic, P_subspaced,
                            &target_factor_subspaced, &target_factor,
                            &other_factor_subspaced, &cursor, &X_compressed,
                            &config, solver_config, &predictions]() {
        Real P_local;
        Real vcache;
        Real B;
        Real observation_bias =
            config.loss_type == LossType::IALSPP ? 0.0 : config.alpha0;

        while (true) {
          int cursor_local = cursor.fetch_add(1);
          if (cursor_local >= target_factor_subspaced.rows()) {
            break;
          }

          P_local = P_quadratic;

          const auto inner_cursor_start =
              X_compressed.outerIndexPtr()[cursor_local];
          auto inner_cursor_end =
              X_compressed.outerIndexPtr()[cursor_local + 1];
          auto nnz = inner_cursor_end - inner_cursor_start;
          auto reg =
              this->compute_reg(nnz, other_factor_subspaced.rows(), config);

          B = P_subspaced.dot(target_factor.row(cursor_local));
          B += reg * target_factor_subspaced(cursor_local);

          int64_t inner_cursor = inner_cursor_start;

          for (SparseMatrix::InnerIterator it(X_compressed, cursor_local); it;
               ++it) {
            int64_t other_index = it.col();
            vcache = other_factor_subspaced(other_index);
            Real residual = (it.value() * (predictions(inner_cursor++) - 1) -
                             observation_bias);
            P_local += it.value() * vcache * vcache;
            B += residual * vcache;
          }
          P_local += reg;

          // difference. Current - vcache = newvector
          vcache = B / P_local;
          target_factor_subspaced(cursor_local) -= vcache;

          inner_cursor = inner_cursor_start;
          for (SparseMatrix::InnerIterator it(X_compressed, cursor_local); it;
               ++it) {
            int64_t other_index = it.col();
            predictions.coeffRef(inner_cursor++) -=
                vcache * (other_factor_subspaced(other_index));
          }
        }
      });
    }
    for (auto &worker : workers) {
      worker.join();
    }
    target_factor.col(dim_start) = target_factor_subspaced;
  }

public:
  inline void step_with_prior(DenseMatrix &target_factor,
                              const SparseMatrix &X,
                              const DenseMatrix &other_factor,
                              const DenseMatrix &prior,
                              const IALSModelConfig &config,
                              const SolverConfig &solver_config) const {
    if (solver_config.solver_type == SolverType::CG) {
      step_cg_with_prior(target_factor, X, other_factor, prior, config,
                         solver_config);
    } else if (solver_config.solver_type == SolverType::Cholesky) {
      step_cholesky_with_prior(target_factor, X, other_factor, prior, config,
                               solver_config);
    } else {
      throw std::invalid_argument(
          "Feature-aware iALS does not support IALSPP.");
    }
  }
  inline void step(DenseMatrix &target_factor, const SparseMatrix &X,
                   const DenseMatrix &other_factor,
                   const IALSModelConfig &config,
                   const SolverConfig &solver_config) const {
    if (solver_config.solver_type == SolverType::CG) {
      step_cg(target_factor, X, other_factor, config, solver_config);
    } else if (solver_config.solver_type == SolverType::Cholesky) {
      step_cholesky(target_factor, X, other_factor, config, solver_config);
    } else {
      if (solver_config.ialspp_subspace_dimension > 1u) {
        step_ialspp(target_factor, X, other_factor, config, solver_config);
      } else {
        step_icd(target_factor, X, other_factor, config, solver_config);
      }
    }
  }
  // DenseMatrix &factor;
  DenseMatrix P;
  DenseMatrix Pinv;
  bool p_initialized;
}; // namespace ials

inline int64_t feature_rows(const FeatureMatrix &features) {
  return std::visit(
      [](const auto &feature) -> int64_t { return feature.rows(); }, features);
}

inline int64_t feature_cols(const FeatureMatrix &features) {
  return std::visit(
      [](const auto &feature) -> int64_t { return feature.cols(); }, features);
}

inline void make_feature_compressed(FeatureMatrix &features) {
  if (auto *sparse_feature = std::get_if<SparseMatrix>(&features)) {
    sparse_feature->makeCompressed();
  }
}

inline DenseMatrix feature_times_weight(const FeatureMatrix &features,
                                        const DenseMatrix &weight) {
  return std::visit(
      [&weight](const auto &feature) -> DenseMatrix { return feature * weight; },
      features);
}

struct IALSTrainer {
  inline IALSTrainer(const IALSModelConfig &config, const SparseMatrix &X)
      : config_(config), K(config.K), n_users(X.rows()), n_items(X.cols()),
        user(n_users, K), item(n_items, K), user_solver(config),
        item_solver(config), X(X), X_t(X.transpose()) {

    this->X.makeCompressed();
    this->X_t.makeCompressed();

    user_solver.initialize(user, config);
    item_solver.initialize(item, config);
  }

  inline IALSTrainer(const IALSModelConfig &config, const SparseMatrix &X,
                     const FeatureMatrix &user_features,
                     const FeatureMatrix &item_features)
      : config_(config), K(config.K), n_users(X.rows()), n_items(X.cols()),
        user(n_users, K), item(n_items, K),
        user_feature_weight(feature_cols(user_features), K),
        item_feature_weight(feature_cols(item_features), K),
        user_solver(config), item_solver(config), X(X), X_t(X.transpose()),
        user_features(user_features), item_features(item_features),
        feature_aware_(true) {
    initialize_feature_aware(feature_rows(user_features),
                             feature_cols(user_features),
                             feature_rows(item_features),
                             feature_cols(item_features),
                             config);
    this->X.makeCompressed();
    this->X_t.makeCompressed();
    make_feature_compressed(this->user_features);
    make_feature_compressed(this->item_features);
    user_solver.initialize(user, config);
    item_solver.initialize(item, config);
  }

  // used when deserialize
  inline IALSTrainer(const IALSModelConfig &config, const DenseMatrix &user_,
                     const DenseMatrix &item_)
      : config_(config), K(user_.cols()), n_users(user_.rows()),
        n_items(item_.rows()), user(user_), item(item_), user_solver(config),
        item_solver(config) {
    const size_t processor_count = std::thread::hardware_concurrency();
    auto solver_config =
        SolverConfig::Builder{}.set_n_threads(processor_count).build();
    this->user_solver.prepare_p(item, config, solver_config);
    this->item_solver.prepare_p(user, config, solver_config);
  }

  inline void step(const SolverConfig &solver_config) {
    if (feature_aware_ && solver_config.solver_type == SolverType::IALSPP)
      throw std::invalid_argument(
          "Feature-aware iALS does not support IALSPP.");
    if (feature_aware_ && epoch_ >= config_.feature_warmup_epochs) {
      user_solver.prepare_p(item, config_, solver_config);
      if (user_feature_weight.rows()) {
        DenseMatrix user_prior = stored_user_feature_prior();
        user_solver.step_with_prior(user, X, item, user_prior, config_,
                                    solver_config);
        update_stored_user_feature_weight();
      } else {
        user_solver.step(user, X, item, config_, solver_config);
      }
      item_solver.prepare_p(user, config_, solver_config);
      if (item_feature_weight.rows()) {
        DenseMatrix item_prior = stored_item_feature_prior();
        item_solver.step_with_prior(item, X_t, user, item_prior, config_,
                                    solver_config);
        update_stored_item_feature_weight();
      } else {
        item_solver.step(item, X_t, user, config_, solver_config);
      }
      ++epoch_;
      return;
    }
    user_solver.prepare_p(item, config_, solver_config);
    user_solver.step(user, X, item, config_, solver_config);
    item_solver.prepare_p(user, config_, solver_config);
    item_solver.step(item, X_t, user, config_, solver_config);
    ++epoch_;
  };

  inline DenseMatrix transform_user(const SparseMatrix &X,
                                    const SolverConfig &solver_config) {
    this->user_solver.prepare_p(item, config_, solver_config);
    return this->user_solver.X_to_vector(X, item, config_, solver_config);
  }

  inline DenseMatrix transform_item(const SparseMatrix &X,
                                    const SolverConfig &solver_config) {
    this->item_solver.prepare_p(user, config_, solver_config);
    return this->item_solver.X_to_vector(X.transpose(), user, config_,
                                         solver_config);
  }

  inline DenseMatrix
  transform_user_with_feature(const SparseMatrix &X,
                              const FeatureMatrix &features,
                              const SolverConfig &solver_config) {
    this->user_solver.prepare_p(item, config_, solver_config);
    DenseMatrix prior = transform_user_feature(features);
    return this->user_solver.X_to_vector_with_prior(X, item, prior, config_,
                                                    solver_config);
  }

  inline DenseMatrix
  transform_item_with_feature(const SparseMatrix &X,
                              const FeatureMatrix &features,
                              const SolverConfig &solver_config) {
    this->item_solver.prepare_p(user, config_, solver_config);
    DenseMatrix prior = transform_item_feature(features);
    return this->item_solver.X_to_vector_with_prior(
        X.transpose(), user, prior, config_, solver_config);
  }

  inline DenseMatrix
  transform_user_feature(const FeatureMatrix &features) const {
    validate_user_feature_matrix(feature_cols(features));
    return feature_times_weight(features, user_feature_weight);
  }

  inline DenseMatrix
  transform_item_feature(const FeatureMatrix &features) const {
    validate_item_feature_matrix(feature_cols(features));
    return feature_times_weight(features, item_feature_weight);
  }

  inline Real compute_loss(const SolverConfig &solver_config) {
    this->user_solver.prepare_p(item, config_, solver_config);
    this->item_solver.prepare_p(user, config_, solver_config);
    Real loss =
        (this->user_solver.P.array() * this->item_solver.P.array()).sum() /
        this->config_.alpha0;
    {
      std::atomic<uint64_t> cursor(0);
      std::vector<std::future<Real>> workers;
      for (size_t i = 0; i < solver_config.n_threads; i++) {

        workers.emplace_back(std::async(std::launch::async, [&cursor, this]() {
          Real loss_local = 0;
          Real observation_bias = this->config_.loss_type == LossType::IALSPP
                                      ? 0.0
                                      : this->config_.alpha0;

          while (true) {
            int cursor_local = cursor.fetch_add(1);
            if (cursor_local >= this->user.rows()) {
              break;
            }
            size_t nnz = 0;
            for (SparseMatrix::InnerIterator it(this->X, cursor_local); it;
                 ++it) {
              nnz++;
              Real prediction =
                  this->user.row(cursor_local).dot(this->item.row(it.col()));
              loss_local += it.value() * prediction * prediction -
                            2 * (it.value() + observation_bias) * prediction +
                            it.value() + observation_bias;
            }
            const Real regularization = this->user_solver.compute_reg(
                nnz, this->item.rows(), this->config_);

            if (!(this->feature_aware_ &&
                  this->user_feature_weight.rows())) {
              loss_local +=
                  regularization * this->user.row(cursor_local).squaredNorm();
            }
          }
          return loss_local;
        }));
      }
      for (auto &w : workers) {
        loss += w.get();
      }
    }
    {
      std::atomic<uint64_t> cursor(0);
      std::vector<std::future<Real>> workers;
      for (size_t i = 0; i < solver_config.n_threads; i++) {

        workers.emplace_back(std::async(std::launch::async, [&cursor, this]() {
          Real loss_local = 0;
          while (true) {
            int cursor_local = cursor.fetch_add(1);
            if (cursor_local >= this->item.rows()) {
              break;
            }
            auto start = this->X_t.outerIndexPtr()[cursor_local];
            auto end = this->X_t.outerIndexPtr()[cursor_local + 1];
            int64_t nnz = end - start;
            const Real regularization = this->item_solver.compute_reg(
                nnz, this->user.rows(), this->config_);

            if (!(this->feature_aware_ &&
                  this->item_feature_weight.rows())) {
              loss_local +=
                  regularization * this->item.row(cursor_local).squaredNorm();
            }
          }
          return loss_local;
        }));
      }
      for (auto &w : workers) {
        loss += w.get();
      }
    }
    if (feature_aware_) {
      if (user_feature_weight.rows()) {
        DenseMatrix residual = user - stored_user_feature_prior();
        for (int64_t row = 0; row < user.rows(); ++row) {
          const int64_t nnz = X.outerIndexPtr()[row + 1] - X.outerIndexPtr()[row];
          loss += user_solver.compute_reg(nnz, item.rows(), config_) *
                  residual.row(row).squaredNorm();
        }
        loss += config_.lambda_user_feature * user_feature_weight.squaredNorm();
      }
      if (item_feature_weight.rows()) {
        DenseMatrix residual = item - stored_item_feature_prior();
        for (int64_t row = 0; row < item.rows(); ++row) {
          const int64_t nnz =
              X_t.outerIndexPtr()[row + 1] - X_t.outerIndexPtr()[row];
          loss += item_solver.compute_reg(nnz, user.rows(), config_) *
                  residual.row(row).squaredNorm();
        }
        loss += config_.lambda_item_feature * item_feature_weight.squaredNorm();
      }
    }
    return loss / 2;
  }

  DenseMatrix user_scores(size_t userblock_begin, size_t userblock_end,
                          const SolverConfig &solver_config) {
    irspack::check_arg(
        userblock_end >= userblock_begin,
        "userblock_end must be greater than or equal to userblock_begin");
    irspack::check_arg(
        n_users >= userblock_end,
        "userblock_end must be smaller than or equal to n_users");

    int64_t result_size = userblock_end - userblock_begin;
    DenseMatrix result(result_size, n_items);
    std::vector<std::thread> workers;
    std::atomic<int64_t> cursor(0);
    for (size_t ind = 0; ind < solver_config.n_threads; ind++) {
      workers.emplace_back([this, userblock_begin, &cursor, result_size,
                            &result]() {
        const int64_t chunk_size = 16;
        while (true) {
          auto block_begin = cursor.fetch_add(chunk_size);
          if (block_begin >= result_size) {
            break;
          }
          auto block_size =
              std::min(block_begin + chunk_size, result_size) - block_begin;
          result.middleRows(block_begin, block_size).noalias() =
              this->user.middleRows(block_begin + userblock_begin, block_size) *
              this->item.transpose();
        }
      });
    }
    for (auto &worker : workers) {
      worker.join();
    }
    return result;
  }

public:
  const IALSModelConfig config_;
  const size_t K;
  const size_t n_users, n_items;
  DenseMatrix user, item;
  DenseMatrix user_feature_weight, item_feature_weight;
  Solver user_solver, item_solver;

private:
  SparseMatrix X, X_t;
  FeatureMatrix user_features, item_features;
  bool feature_aware_ = false;
  size_t epoch_ = 0;

  inline void initialize_feature_aware(size_t user_feature_rows,
                                       size_t user_feature_cols,
                                       size_t item_feature_rows,
                                       size_t item_feature_cols,
                                       const IALSModelConfig &config) {
    if (user_feature_rows != n_users || item_feature_rows != n_items)
      throw std::invalid_argument("Feature matrix row count mismatch.");
    if ((user_feature_cols && config.lambda_user_feature <= 0) ||
        (item_feature_cols && config.lambda_item_feature <= 0))
      throw std::invalid_argument(
          "Feature weight regularization must be positive.");
    user_feature_weight.setZero();
    item_feature_weight.setZero();
  }

  inline void validate_user_feature_matrix(int64_t cols) const {
    if (user_feature_weight.cols() != K) {
      throw std::invalid_argument(
          "User feature weights are not initialized.");
    }
    if (cols != user_feature_weight.rows()) {
      std::stringstream ss;
      ss << "Shape mismatch: user feature matrix has " << cols
         << " columns but user_feature_weight has "
         << user_feature_weight.rows() << " rows.";
      throw std::invalid_argument(ss.str());
    }
  }

  inline void validate_item_feature_matrix(int64_t cols) const {
    if (item_feature_weight.cols() != K) {
      throw std::invalid_argument(
          "Item feature weights are not initialized.");
    }
    if (cols != item_feature_weight.rows()) {
      std::stringstream ss;
      ss << "Shape mismatch: item feature matrix has " << cols
         << " columns but item_feature_weight has "
         << item_feature_weight.rows() << " rows.";
      throw std::invalid_argument(ss.str());
    }
  }

  inline DenseMatrix stored_user_feature_prior() const {
    return feature_times_weight(user_features, user_feature_weight);
  }

  inline DenseMatrix stored_item_feature_prior() const {
    return feature_times_weight(item_features, item_feature_weight);
  }

  inline void update_stored_user_feature_weight() {
    update_feature_weight(user_features, user, user_feature_weight,
                          config_.lambda_user_feature, X, item.rows());
  }

  inline void update_stored_item_feature_weight() {
    update_feature_weight(item_features, item, item_feature_weight,
                          config_.lambda_item_feature, X_t, user.rows());
  }

  inline void update_feature_weight(const FeatureMatrix &features,
                                    const DenseMatrix &factor,
                                    DenseMatrix &weight,
                                    Real lambda_feature,
                                    const SparseMatrix &interactions,
                                    int64_t other_size) {
    std::visit(
        [&](const auto &feature) {
          update_feature_weight(feature, factor, weight, lambda_feature,
                                interactions, other_size);
        },
        features);
  }

  inline void update_feature_weight(const SparseMatrix &features,
                                    const DenseMatrix &factor,
                                    DenseMatrix &weight,
                                    Real lambda_feature,
                                    const SparseMatrix &interactions,
                                    int64_t other_size) {
    if (features.cols() == 0)
      return;
    SparseMatrix weighted_features = features;
    DenseMatrix weighted_factor = factor;
    for (int64_t row = 0; row < features.rows(); ++row) {
      const int64_t nnz = interactions.outerIndexPtr()[row + 1] -
                          interactions.outerIndexPtr()[row];
      const Real row_weight =
          user_solver.compute_reg(nnz, other_size, config_);
      for (SparseMatrix::InnerIterator it(weighted_features, row); it; ++it)
        it.valueRef() *= std::sqrt(row_weight);
      weighted_factor.row(row) *= row_weight;
    }
    DenseMatrix gram =
        DenseMatrix(weighted_features.transpose() * weighted_features);
    gram.diagonal().array() += lambda_feature;
    DenseMatrix rhs = features.transpose() * weighted_factor;
    Eigen::LLT<DenseMatrix> llt(gram);
    if (llt.info() != Eigen::Success)
      throw std::runtime_error("Feature ridge Cholesky decomposition failed.");
    weight = llt.solve(rhs);
  }

  inline void update_feature_weight(const DenseMatrix &features,
                                    const DenseMatrix &factor,
                                    DenseMatrix &weight,
                                    Real lambda_feature,
                                    const SparseMatrix &interactions,
                                    int64_t other_size) {
    if (features.cols() == 0)
      return;
    DenseMatrix weighted_features = features;
    DenseMatrix weighted_factor = factor;
    for (int64_t row = 0; row < features.rows(); ++row) {
      const int64_t nnz = interactions.outerIndexPtr()[row + 1] -
                          interactions.outerIndexPtr()[row];
      const Real row_weight =
          user_solver.compute_reg(nnz, other_size, config_);
      weighted_features.row(row) *= std::sqrt(row_weight);
      weighted_factor.row(row) *= row_weight;
    }
    DenseMatrix gram = weighted_features.transpose() * weighted_features;
    gram.diagonal().array() += lambda_feature;
    DenseMatrix rhs = features.transpose() * weighted_factor;
    Eigen::LLT<DenseMatrix> llt(gram);
    if (llt.info() != Eigen::Success)
      throw std::runtime_error("Feature ridge Cholesky decomposition failed.");
    weight = llt.solve(rhs);
  }
};
} // namespace ials
} // namespace irspack
