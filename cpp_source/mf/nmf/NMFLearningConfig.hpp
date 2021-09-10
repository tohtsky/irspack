#pragma once

#include "../definitions.hpp"
#include <cstddef>
#include <map>
#include <vector>

namespace irspack {
namespace mf {
namespace nmf {
using namespace std;
struct NMFLearningConfig {
  inline NMFLearningConfig(size_t K, Real l2_reg, Real l1_reg, int random_seed,
                           size_t n_threads, bool shuffle)
      : K(K), l2_reg(l2_reg), l1_reg(l1_reg), random_seed(random_seed),
        n_threads(n_threads) {}

  NMFLearningConfig(const NMFLearningConfig &other) = default;

  const size_t K;
  const Real l2_reg, l1_reg;
  int random_seed;
  size_t n_threads;
  bool shuffle;

  struct Builder {
    Real reg = .1;
    Real l2_reg = 0;
    Real l1_reg = 0;
    size_t K = 16;
    int random_seed = 42;
    size_t n_threads = 1;
    bool shuffle = true;
    inline Builder() {}
    inline NMFLearningConfig build() {
      return NMFLearningConfig(K, l2_reg, l1_reg, random_seed, n_threads,
                               shuffle);
    }

    Builder &set_l2_reg(Real l2_reg) {
      this->l2_reg = l2_reg;
      return *this;
    }

    Builder &set_l1_reg(Real l1_reg) {
      this->l1_reg = l1_reg;
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
    Builder &set_shuffle(bool shuffle) {
      this->shuffle = shuffle;
      return *this;
    }
  };
};
} // namespace nmf
} // namespace mf
} // namespace irspack
