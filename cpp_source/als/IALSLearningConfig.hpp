#pragma once

#include "definitions.hpp"
#include <cstddef>
#include <map>
#include <vector>

namespace irspack {
namespace ials {

enum class LossType { Original, IALSPP };
using namespace std;

struct IALSModelConfig {
  inline IALSModelConfig(size_t K, Real alpha0, Real reg, Real nu,
                         Real init_stdev, int random_seed, LossType loss_type)
      : K(K), alpha0(alpha0), reg(reg), nu(nu), init_stdev(init_stdev),
        random_seed(random_seed), loss_type(loss_type) {}
  const size_t K;
  const Real alpha0, reg, nu, init_stdev;
  int random_seed;
  LossType loss_type;

  struct Builder {
    Real reg = .1;
    Real alpha0 = 0.1;
    Real nu = 1.0;
    Real init_stdev = .1;
    size_t K = 16;
    int random_seed = 42;
    LossType loss_type = LossType::IALSPP;
    inline Builder() {}
    inline IALSModelConfig build() {
      return IALSModelConfig(K, alpha0, reg, nu, init_stdev, random_seed,
                             loss_type);
    }

    Builder &set_alpha0(Real alpha0) {
      this->alpha0 = alpha0;
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
    Builder &set_nu(Real nu) {
      this->nu = nu;
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
    Builder &set_loss_type(LossType loss_type) {
      this->loss_type = loss_type;
      return *this;
    }
  };
};

struct SolverConfig {

  inline SolverConfig(size_t n_threads, bool use_cg, size_t max_cg_steps)
      : n_threads(n_threads), use_cg(use_cg), max_cg_steps(max_cg_steps) {}

  SolverConfig(const SolverConfig &other) = default;

  size_t n_threads;
  bool use_cg;
  size_t max_cg_steps;
  struct Builder {
    size_t n_threads = 1;
    bool use_cg = true;
    size_t max_cg_steps = 0;

    inline Builder() {}
    inline SolverConfig build() {
      return SolverConfig(n_threads, use_cg, max_cg_steps);
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
} // namespace ials
} // namespace irspack
