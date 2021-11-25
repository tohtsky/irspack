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
#include <vector>

namespace irspack {
namespace ials {
using namespace std;
// using SymmetricMatrix = Eigen::SelfAdjointView<DenseMatrix &, Eigen::Upper>;
using SymmetricMatrix =
    decltype(DenseMatrix{0, 0}.selfadjointView<Eigen::Upper>());
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

public:
  inline void step(DenseMatrix &target_factor, const SparseMatrix &X,
                   const DenseMatrix &other_factor,
                   const IALSModelConfig &config,
                   const SolverConfig &solver_config) const {
    if (solver_config.solver_type == SolverType::CG) {
      step_cg(target_factor, X, other_factor, config, solver_config);
    } else if (solver_config.solver_type == SolverType::Cholesky) {
      step_cholesky(target_factor, X, other_factor, config, solver_config);
    } else {
      step_ialspp(target_factor, X, other_factor, config, solver_config);
    }
  }
  // DenseMatrix &factor;
  DenseMatrix P;
  DenseMatrix Pinv;
  bool p_initialized;
}; // namespace ials

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

    user_solver.prepare_p(item, config_, solver_config);
    user_solver.step(user, X, item, config_, solver_config);
    item_solver.prepare_p(user, config_, solver_config);
    item_solver.step(item, X_t, user, config_, solver_config);
  };

  inline DenseMatrix transform_user(const SparseMatrix &X,
                                    const SolverConfig &solver_config) const {
    return this->user_solver.X_to_vector(X, item, config_, solver_config);
  }

  inline DenseMatrix transform_item(const SparseMatrix &X,
                                    const SolverConfig &solver_config) const {
    return this->item_solver.X_to_vector(X.transpose(), user, config_,
                                         solver_config);
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
  Solver user_solver, item_solver;

private:
  SparseMatrix X, X_t;
};
} // namespace ials
} // namespace irspack
