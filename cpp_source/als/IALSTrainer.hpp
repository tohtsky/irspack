#pragma once
#include "IALSLearningConfig.hpp"
#include "definitions.hpp"
#include <Eigen/Cholesky>
#include <Eigen/IterativeLinearSolvers>
#include <atomic>
#include <cstddef>
#include <future>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ials11 {
using namespace std;

struct Solver {
  Solver(size_t K, Real reg) : reg(reg), P(K, K) {}

  inline void initialize(DenseMatrix &factor, Real init_stdev,
                         int random_seed) {
    if (init_stdev > 0) {
      std::mt19937 gen(random_seed);
      std::normal_distribution<Real> dist(0.0, init_stdev);
      for (int i = 0; i < factor.rows(); i++) {
        for (int k = 0; k < factor.cols(); k++) {
          factor(i, k) = dist(gen);
        }
      }
    }
  }

  inline void prepare_p(const DenseMatrix &other_factor,
                        const IALSLearningConfig &config) {
    const int64_t mb_size = 1020;
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
    for (int i = 0; i < P.rows(); i++) {
      P.coeffRef(i, i) += this->reg;
    }
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
          for (SparseMatrix::InnerIterator it(X, cursor_local); it; ++it) {
            b.noalias() += (it.value() * (1 + config.alpha)) *
                           other_factor.row(it.col()).transpose();
          }

          r = b - P * x;
          size_t nnz = 0;
          for (SparseMatrix::InnerIterator it(X, cursor_local); it; ++it) {
            Real vdotx = other_factor.row(it.col()) * x;
            r.noalias() -= (it.value() * config.alpha * vdotx) *
                           other_factor.row(it.col()).transpose();
            nnz++;
          }
          if (nnz == 0u) {
            target_factor.row(cursor_local).array() = 0;
            continue;
          }
          p = r;

          size_t cg_max_iter =
              config.max_cg_steps == 0u ? P.rows() : config.max_cg_steps;

          for (size_t cg_iter = 0; cg_iter < cg_max_iter; cg_iter++) {
            Real r2 = r.squaredNorm();
            Ap = P * p;
            for (SparseMatrix::InnerIterator it(X, cursor_local); it; ++it) {
              Real vdotp = other_factor.row(it.col()) * p;
              Ap += (it.value() * config.alpha * vdotp) *
                    other_factor.row(it.col()).transpose();
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
          for (SparseMatrix::InnerIterator it(X, cursor_local); it; ++it) {
            std::int64_t other_index = it.col();
            Real alphaX = (config.alpha * it.value());
            vcache = other_factor.row(other_index).transpose();
            P_local.noalias() += alphaX * vcache * vcache.transpose();
            B.noalias() += (it.value() * (1 + config.alpha)) *
                           other_factor.row(other_index).transpose();
          }
          Eigen::ConjugateGradient<DenseMatrix> cg;
          cg.compute(P);
          target_factor.row(cursor_local) =
              cg.solveWithGuess(B, target_factor.row(cursor_local).transpose());
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
  Real reg;
  // DenseMatrix &factor;
  DenseMatrix P;
  DenseMatrix Pinv;
}; // namespace ials11

struct IALSTrainer {
  inline IALSTrainer(const IALSLearningConfig &config, const SparseMatrix &X)
      : config_(config), K(config.K), n_users(X.rows()), n_items(X.cols()),
        user(n_users, K), item(n_items, K), user_solver(K, config.reg),
        item_solver(K, config.reg), X(X), X_t(X.transpose()) {
    this->X.makeCompressed();
    this->X_t.makeCompressed();

    user_solver.initialize(user, config.init_stdev, config.random_seed);
    item_solver.initialize(item, config.init_stdev, config.random_seed);
  }

  inline IALSTrainer(const IALSLearningConfig &config, const DenseMatrix &user_,
                     const DenseMatrix &item_)
      : config_(config), K(user_.cols()), n_users(user_.rows()),
        n_items(item_.rows()), user(user_), item(item_),
        user_solver(K, config.reg), item_solver(K, config.reg) {
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
    if (userblock_end < userblock_begin) {
      throw std::invalid_argument("begin > end");
    }
    const size_t result_size = userblock_end - userblock_begin;
    if (userblock_end > n_users) {
      throw std::invalid_argument("end > n_users");
    }
    DenseMatrix result(result_size, n_items);
    size_t chunked_blocksize = result_size / config_.n_threads;
    size_t remnant = result_size % config_.n_threads;
    std::vector<std::thread> workers;
    size_t block_begin = userblock_begin;
    for (size_t ind = 0; ind < config_.n_threads; ind++) {
      size_t block_size = chunked_blocksize;
      if (ind < remnant) {
        block_size++;
      }
      workers.emplace_back([this, block_begin, userblock_begin, block_size,
                            &result]() {
        result.middleRows(block_begin - userblock_begin, block_size).noalias() =
            this->user.middleRows(block_begin, block_size) *
            this->item.transpose();
      });
      block_begin += block_size;
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
} // namespace ials11
