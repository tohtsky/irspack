#pragma once

#include "definitions.hpp"
#include <cstddef>
#include <map>
#include <vector>

namespace ials11 {
using namespace std;

struct IALSLearningConfig {
  inline IALSLearningConfig(size_t K, Real alpha, Real reg, Real init_stdev,
                            int random_seed, size_t n_threads, bool use_cg,
                            size_t max_cg_steps)
      : K(K), alpha(alpha), reg(reg), init_stdev(init_stdev),
        random_seed(random_seed), n_threads(n_threads), use_cg(use_cg),
        max_cg_steps(max_cg_steps) {}

  IALSLearningConfig(const IALSLearningConfig &other) = default;

  const size_t K;
  const Real alpha, reg, init_stdev;
  int random_seed;
  size_t n_threads;
  bool use_cg;
  size_t max_cg_steps;

  struct Builder {
    Real reg = .1;
    Real alpha = 10;
    Real init_stdev = .1;
    size_t K = 16;
    int random_seed = 42;
    size_t n_threads = 1;
    bool use_cg = true;
    size_t max_cg_steps = 0;
    inline Builder() {}
    inline IALSLearningConfig build() {
      return IALSLearningConfig(K, alpha, reg, init_stdev, random_seed,
                                n_threads, use_cg, max_cg_steps);
    }

    Builder &set_alpha(Real alpha) {
      this->alpha = alpha;
      return *this;
    }

    Builder &set_init_stdev(Real init_stdev) {
      this->init_stdev = init_stdev;
      return *this;
    }
    Builder &set_reg(Real reg) {
      this->reg = reg;
      return *this;
    }

    Builder &set_K(size_t K) {
      this->K = K;
      return *this;
    }

    Builder &set_random_seed(int random_seed) {
      this->random_seed = random_seed;
      return *this;
    }

    Builder &set_n_threads(size_t n_threads) {
      this->n_threads = n_threads;
      return *this;
    }
    Builder &set_use_cg(bool use_cg) {
      this->use_cg = use_cg;
      return *this;
    }
    Builder &set_max_cg_steps(size_t max_cg_steps) {
      this->max_cg_steps = max_cg_steps;
      return *this;
    }
  };
};
} // namespace ials11
