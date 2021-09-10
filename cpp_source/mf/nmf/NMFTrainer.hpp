#pragma once
#include "../../argcheck.hpp"
#include "../definitions.hpp"
#include "NMFLearningConfig.hpp"
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
namespace mf {
namespace nmf {
using namespace std;

struct Solver {
  Solver(size_t K, Real l2_reg, Real l1_reg)
      : l2_reg(l2_reg), l1_reg(l1_reg), P(K, K) {}

  inline void initialize(DenseMatrix &factor, Real scale, int random_seed) {
    if (scale > 0) {
      std::mt19937 gen(random_seed);
      std::normal_distribution<Real> dist(0.0, scale);
      for (int i = 0; i < factor.rows(); i++) {
        for (int k = 0; k < factor.cols(); k++) {
          factor(i, k) = std::abs(dist(gen));
        }
      }
    }
  }

  inline void prepare_p(const DenseMatrix &other_factor, const SparseMatrix &Xt,
                        const NMFLearningConfig &config) {
    const int64_t mb_size = 16;
    P = DenseMatrix::Zero(other_factor.cols(), other_factor.cols());
    XH = DenseMatrix(Xt.cols(), other_factor.cols());
    XH.array() = -this->l1_reg;

    std::atomic<int64_t> cursor{static_cast<size_t>(0)};

    std::vector<std::future<std::pair<DenseMatrix, DenseMatrix>>> workers;
    for (size_t i = 0; i < config.n_threads; i++) {
      workers.emplace_back(std::async(std::launch::async, [&other_factor, &Xt,
                                                           &cursor, mb_size]() {
        DenseMatrix P_local =
            DenseMatrix::Zero(other_factor.cols(), other_factor.cols());
        DenseMatrix XH_local =
            DenseMatrix::Zero(Xt.cols(), other_factor.cols());

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
          XH_local.noalias() +=
              Xt.middleRows(cursor_local, block_end - cursor_local)
                  .transpose() *
              other_factor.middleRows(cursor_local, block_end - cursor_local);
        }
        std::pair<DenseMatrix, DenseMatrix> result = {std::move(P_local),
                                                      std::move(XH_local)};
        return result;
      }));
    }
    for (auto &w : workers) {
      auto thread_result = w.get();
      P.noalias() += thread_result.first;
      XH.noalias() += thread_result.second;
    }
    for (int i = 0; i < P.rows(); i++) {
      P.coeffRef(i, i) += this->l2_reg;
    }
  }

  inline DenseMatrix X_to_vector(const SparseMatrix &X,
                                 const DenseMatrix &other_factor,
                                 const NMFLearningConfig &config) const {
    if (X.cols() != other_factor.rows()) {
      std::stringstream ss;
      ss << "Shape mismatch: X.cols() = " << X.cols()
         << " but other.factor.rows() = " << other_factor.rows() << ".";
      throw std::invalid_argument(ss.str());
    }
    DenseMatrix result = DenseMatrix::Zero(X.rows(), P.rows());
    this->step(result, X, other_factor, config);

    return result;
  }

  inline Real step(DenseMatrix &target_factor, const SparseMatrix &X,
                   const DenseMatrix &other_factor,
                   const NMFLearningConfig &config) const {

    Real violation = 0;
    for (size_t t = 0; t < config.K; t++) {
      std::atomic<int> cursor(0);
      std::vector<std::future<Real>> workers;

      Real hessian = this->P(t, t);
      for (size_t ind = 0; ind < config.n_threads; ind++) {
        workers.emplace_back(
            std::async([this, t, hessian, &target_factor, &cursor, &violation,
                        &X, &other_factor, &config]() {
              Real violation_local = 0;
              constexpr Real ZERO = 0;
              while (true) {
                int i = cursor.fetch_add(1);
                if (i >= target_factor.rows()) {
                  break;
                }
                Real gradient = -this->XH(i, t);
                gradient += P.row(t) * target_factor.row(i).transpose();
                Real projected_gradient = target_factor(i, t) == ZERO
                                              ? std::min(ZERO, gradient)
                                              : gradient;
                violation_local += projected_gradient;
                if (hessian != ZERO) {
                  target_factor(i, t) =
                      std::max(target_factor(i, t) - gradient / hessian, ZERO);
                }
              }
              return violation_local;
            }));
      }
      for (auto &w : workers) {
        violation += w.get();
      }
    }
    return violation;
  }

  Real l2_reg, l1_reg;
  // DenseMatrix &factor;
  DenseMatrix P;
  DenseMatrix XH;
}; // namespace nmf

struct NMFTrainer {
  inline NMFTrainer(const NMFLearningConfig &config, const SparseMatrix &X)
      : config_(config), K(config.K), n_users(X.rows()), n_items(X.cols()),
        user(n_users, K), item(n_items, K),
        user_solver(K, config.l2_reg, config.l1_reg),
        item_solver(K, config.l2_reg, config.l1_reg), X(X), X_t(X.transpose()) {
    this->X.makeCompressed();
    this->X_t.makeCompressed();
    Real avg =
        static_cast<Real>(this->X.sum()) / static_cast<Real>(n_users) / n_items;
    user_solver.initialize(user, avg, config.random_seed);
    item_solver.initialize(item, avg, config.random_seed);
  }

  inline NMFTrainer(const NMFLearningConfig &config, const DenseMatrix &user_,
                    const DenseMatrix &item_, const SparseMatrix X)
      : config_(config), K(user_.cols()), n_users(user_.rows()),
        n_items(item_.rows()), user(user_), item(item_),
        user_solver(K, config.l2_reg, config.l1_reg),
        item_solver(K, config.l2_reg, config.l1_reg), X(X), X_t(X.transpose()) {
    this->user_solver.prepare_p(item, X_t, config);
    this->item_solver.prepare_p(user, X, config);
  }

  inline void step() {

    user_solver.prepare_p(item, X_t, config_);
    user_solver.step(user, X, item, config_);
    item_solver.prepare_p(user, X, config_);
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
  const NMFLearningConfig config_;
  const size_t K;
  const size_t n_users, n_items;
  DenseMatrix user, item;
  Solver user_solver, item_solver;

  SparseMatrix X, X_t;
};
} // namespace nmf
} // namespace mf
} // namespace irspack
