/**
 * Copyright 2020 BizReach, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "IALSLearningConfig.hpp"
#include "definitions.hpp"
#include <Eigen/Cholesky>
#include <atomic>
#include <future>
#include <iostream>
#include <random>
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

  inline void prepare_p(const DenseMatrix &other_factor) {
    P = other_factor.transpose() * other_factor;
    for (int i = 0; i < P.rows(); i++) {
      P(i, i) += this->reg;
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
    DenseMatrix result(X.rows(), P.rows());
    DenseMatrix P_local(P.rows(), P.cols());
    DenseVector B(P.rows());
    for (int i = 0; i < X.rows(); i++) {
      P_local.array() = P.array();
      B.array() = static_cast<Real>(0);
      for (SparseMatrix::InnerIterator it(X, i); it; ++it) {
        P_local += (config.alpha * it.value()) *
                   other_factor.row(it.col()).transpose() *
                   other_factor.row(it.col());

        B += (1 + config.alpha) * other_factor.row(it.col()).transpose();
      }
      Eigen::LLT<Eigen::Ref<DenseMatrix>, Eigen::Upper> U(P_local);
      result.row(i) = U.solve(B).transpose();
    }
    return result;
  }

  inline void step(DenseMatrix &target_factor, const SparseMatrix &X,
                   const DenseMatrix &other_factor,
                   const IALSLearningConfig &config) {
    prepare_p(other_factor);
    std::vector<std::thread> workers;

    std::atomic<int> cursor(0);
    for (size_t ind = 0; ind < config.n_threads; ind++) {
      workers.emplace_back([this, &target_factor, &cursor, &X, &other_factor,
                            &config]() {
        DenseMatrix P_local(P.rows(), P.cols());
        DenseVector B(P.rows());
        while (true) {
          int cursor_local = cursor.fetch_add(1);
          if (cursor_local >= target_factor.rows()) {
            break;
          }
          P_local.array() = P.array();
          B.array() = static_cast<Real>(0);
          for (SparseMatrix::InnerIterator it(X, cursor_local); it; ++it) {
            P_local += (config.alpha * it.value()) *
                       other_factor.row(it.col()).transpose() *
                       other_factor.row(it.col());

            B += (1 + config.alpha) * other_factor.row(it.col()).transpose();
          }
          Eigen::LLT<Eigen::Ref<DenseMatrix>, Eigen::Upper> U(P_local);
          target_factor.row(cursor_local) = U.solve(B).transpose();
        }
      });
    }
    for (auto &w : workers) {
      w.join();
    }
  }
  Real reg;
  // DenseMatrix &factor;
  DenseMatrix P;
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
    this->user_solver.prepare_p(item);
    this->item_solver.prepare_p(user);
  }

  inline void step() {
    user_solver.step(user, X, item, config_);
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
      workers.emplace_back(
          [this, block_begin, userblock_begin, block_size, &result]() {
            result.block(block_begin - userblock_begin, 0, block_size,
                         this->n_items) =
                this->user.block(block_begin, 0, block_size, this->K) *
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