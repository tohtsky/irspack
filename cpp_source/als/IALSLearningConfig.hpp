#pragma once

#include "definitions.hpp"
#include <cstddef>
#include <map>
#include <vector>

namespace irspack {
namespace ials {

enum class LossType { ORIGINAL, IALSPP };
enum class SolverType { Cholesky, CG, IALSPP };
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

  inline SolverConfig(size_t n_threads, SolverType solver_type,
                      size_t max_cg_steps, size_t ialspp_subspace_dimension,
                      size_t ialspp_iteration)
      : n_threads(n_threads), solver_type(solver_type),
        max_cg_steps(max_cg_steps),
        ialspp_subspace_dimension(ialspp_subspace_dimension),
        ialspp_iteration(ialspp_iteration) {}

  SolverConfig(const SolverConfig &other) = default;

  size_t n_threads;
  SolverType solver_type;
  size_t max_cg_steps;
  size_t ialspp_subspace_dimension, ialspp_iteration;

  struct Builder {
    size_t n_threads = 1;
    SolverType solver_type = SolverType::CG;
    size_t max_cg_steps = 3;

    size_t ialspp_subspace_dimension = 64;
    size_t ialspp_iteration = 1;
    inline Builder() {}
    inline SolverConfig build() {
      return SolverConfig(n_threads, solver_type, max_cg_steps,
                          ialspp_subspace_dimension, ialspp_iteration);
    }

    Builder &set_n_threads(size_t n_threads) {
      this->n_threads = n_threads;
      return *this;
    }
    Builder &set_solver_type(SolverType solver_type) {
      this->solver_type = solver_type;
      return *this;
    }
    Builder &set_max_cg_steps(size_t max_cg_steps) {
      this->max_cg_steps = max_cg_steps;
      return *this;
    }
    Builder &set_ialspp_subspace_dimension(size_t ialspp_subspace_dimension) {
      this->ialspp_subspace_dimension = ialspp_subspace_dimension;
      return *this;
    }
    Builder &set_ialspp_iteration(size_t ialspp_iteration) {
      this->ialspp_iteration = ialspp_iteration;
      return *this;
    }
  };
};
} // namespace ials
} // namespace irspack
