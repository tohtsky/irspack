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

#include "definitions.hpp"
#include <map>
#include <vector>

namespace ials11 {
using namespace std;

struct IALSLearningConfig {
  inline IALSLearningConfig(size_t K, Real alpha, Real reg, Real init_stdev,
                            int random_seed, size_t n_threads)
      : K(K), alpha(alpha), reg(reg), init_stdev(init_stdev),
        random_seed(random_seed), n_threads(n_threads) {}

  IALSLearningConfig(const IALSLearningConfig &other) = default;

  const size_t K;
  const Real alpha, reg, init_stdev;
  int random_seed;
  size_t n_threads;

  struct Builder {
    Real reg = .1;
    Real alpha = 10;
    Real init_stdev = .1;
    size_t K = 16;
    int random_seed = 42;
    size_t n_threads = 1;
    inline Builder() {}
    inline IALSLearningConfig build() {
      return IALSLearningConfig(K, alpha, reg, init_stdev, random_seed,
                                n_threads);
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
  };
};
} // namespace ials11
