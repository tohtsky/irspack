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
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace irspack {
namespace ials {
using namespace std;

struct Solver {
  Solver(size_t K) : P(K, K) {}

  inline void initialize(DenseMatrix &factor, Real init_stdev,
                         int random_seed) {

    if (init_stdev > 0) {
      std::mt19937 gen(random_seed);
      std::normal_distribution<Real> dist(0.0, init_stdev /
                                                   std::sqrt(factor.cols()));
      for (int i = 0; i < factor.rows(); i++) {
        for (int k = 0; k < factor.cols(); k++) {
          factor(i, k) = dist(gen);
        }
      }
    }
  }

  inline void prepare_p(const DenseMatrix &other_factor,
                        const IALSLearningConfig &config) {
    const int64_t mb_size = 16;
    P = DenseMatrix::Zero(other_factor.cols(), other_factor.cols());

    std::atomic<int64_t> cursor{static_cast<size_t>(0)};

    std::vector<std::future<DenseMatrix>> workers;
    for (size_t i = 0; i < config.n_threads; i++) {
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
    P *= config.alpha0;
  }

  inline Real compute_reg(const int64_t nnz, const int64_t other_size,
                          const IALSLearningConfig &config) const {
    return config.reg * std::pow(config.alpha0 * other_size + nnz, config.nu);
  }

  inline DenseMatrix X_to_vector(const SparseMatrix &X,
                                 const DenseMatrix &other_factor,
                                 const IALSLearningConfig &config) const {
    if (X.cols() != other_factor.rows()) {
      std::cout << this << std::endl;
      std::stringstream ss;
      ss << "Shape mismatch: X.cols() = " << X.cols()
         << " but other.factor.rows() = " << other_factor.rows() << ".";
      throw std::invalid_argument(ss.str());
    }
    DenseMatrix result = DenseMatrix::Zero(X.rows(), P.rows());
    this->step(result, X, other_factor, config);

    return result;
  }

  inline void step_cg(DenseMatrix &target_factor, const SparseMatrix &X,
                      const DenseMatrix &other_factor,
                      const IALSLearningConfig &config) const {

    std::vector<std::thread> workers;

    std::atomic<int> cursor(0);
    for (size_t ind = 0; ind < config.n_threads; ind++) {
      workers.emplace_back([this, &target_factor, &cursor, &X, &other_factor,
                            &config]() {
        DenseVector b(P.rows()), x(P.rows()), r(P.rows()), p(P.rows()),
            Ap(P.rows());
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
            b.noalias() +=
                ((config.loss_type == LossType::IALSPP ? 0.0 : config.alpha0) +
                 it.value()) *
                other_factor.row(it.col()).transpose();
            nnz++;
          }

          const Real regularization_this =
              this->compute_reg(nnz, other_factor.cols(), config);
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

          size_t cg_max_iter =
              config.max_cg_steps == 0u ? P.rows() : config.max_cg_steps;

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
                            const IALSLearningConfig &config) const {

    std::vector<std::thread> workers;

    std::atomic<int> cursor(0);
    for (size_t ind = 0; ind < config.n_threads; ind++) {
      workers.emplace_back([this, &target_factor, &cursor, &X, &other_factor,
                            &config]() {
        DenseMatrix P_local(P.rows(), P.cols());
        DenseVector B(P.rows());
        DenseVector vcache(P.rows()), r(P.rows());
        while (true) {
          int cursor_local = cursor.fetch_add(1);
          if (cursor_local >= target_factor.rows()) {
            break;
          }
          P_local.noalias() = P;

          B.array() = static_cast<Real>(0);
          int64_t nnz = 0;
          for (SparseMatrix::InnerIterator it(X, cursor_local); it; ++it) {
            int64_t other_index = it.col();
            vcache = other_factor.row(other_index).transpose();
            P_local.noalias() += it.value() * vcache * vcache.transpose();
            B.noalias() +=
                ((config.loss_type == LossType::IALSPP ? 0.0 : config.alpha0) +
                 it.value()) *
                vcache;
            nnz++;
          }
          const Real regularization_this =
              this->compute_reg(nnz, other_factor.cols(), config);

          for (int64_t i = 0; i < P.rows(); i++) {
            P_local(i, i) += regularization_this;
          }

          Eigen::LLT<Eigen::Ref<DenseMatrix>> llt(P_local);
          target_factor.row(cursor_local) = llt.solve(B);
        }
      });
    }
    for (auto &w : workers) {
      w.join();
    }
  }

  inline void step(DenseMatrix &target_factor, const SparseMatrix &X,
                   const DenseMatrix &other_factor,
                   const IALSLearningConfig &config) const {
    if (config.use_cg) {
      step_cg(target_factor, X, other_factor, config);
    } else {
      step_cholesky(target_factor, X, other_factor, config);
    }
  }
  // DenseMatrix &factor;
  DenseMatrix P;
  DenseMatrix Pinv;
}; // namespace irspack

struct IALSTrainer {
  inline IALSTrainer(const IALSLearningConfig &config, const SparseMatrix &X)
      : config_(config), K(config.K), n_users(X.rows()), n_items(X.cols()),
        user(n_users, K), item(n_items, K), user_solver(K), item_solver(K),
        X(X), X_t(X.transpose()) {
    this->X.makeCompressed();
    this->X_t.makeCompressed();

    user_solver.initialize(user, config.init_stdev, config.random_seed);
    item_solver.initialize(item, config.init_stdev, config.random_seed);
  }

  inline IALSTrainer(const IALSLearningConfig &config, const DenseMatrix &user_,
                     const DenseMatrix &item_)
      : config_(config), K(user_.cols()), n_users(user_.rows()),
        n_items(item_.rows()), user(user_), item(item_), user_solver(K),
        item_solver(K) {
    this->user_solver.prepare_p(item, config);
    this->item_solver.prepare_p(user, config);
  }

  inline void step() {

    user_solver.prepare_p(item, config_);
    user_solver.step(user, X, item, config_);
    item_solver.prepare_p(user, config_);
    item_solver.step(item, X_t, user, config_);
  };

  inline DenseMatrix transform_user(const SparseMatrix &X) const {
    return this->user_solver.X_to_vector(X, item, config_);
  }

  inline DenseMatrix transform_item(const SparseMatrix &X) const {
    return this->item_solver.X_to_vector(X.transpose(), user, config_);
  }

  DenseMatrix user_scores(size_t userblock_begin, size_t userblock_end) {
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
    for (size_t ind = 0; ind < config_.n_threads; ind++) {
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
  const IALSLearningConfig config_;
  const size_t K;
  const size_t n_users, n_items;
  DenseMatrix user, item;
  Solver user_solver, item_solver;

private:
  SparseMatrix X, X_t;
};
} // namespace ials
} // namespace irspack
