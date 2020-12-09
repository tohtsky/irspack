"""
Copyright 2020 BizReach, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


from .. import parameter_tuning
from ..definitions import InteractionMatrix
from ._rwr import RandomWalkGenerator
from .base import BaseRecommenderWithThreadingSupport, BaseSimilarityRecommender


class RandomWalkWithRestartRecommender(
    BaseRecommenderWithThreadingSupport, BaseSimilarityRecommender
):
    default_tune_range = [
        parameter_tuning.UniformSuggestion("decay", 1e-2, 9.9e-1),
        parameter_tuning.IntegerSuggestion("n_samples", 100, 2000, step=100),
        parameter_tuning.IntegerSuggestion("cutoff", 100, 2000, step=100),
    ]

    def __init__(
        self,
        X_all: InteractionMatrix,
        decay: float = 0.3,
        cutoff: int = 1000,
        n_samples: int = 1000,
        random_seed: int = 42,
        n_thread: int = 4,
    ):
        super().__init__(X_all, n_thread=n_thread)
        self.decay = decay
        self.n_samples = n_samples
        self.cutoff = cutoff
        self.random_seed = random_seed

    def learn(self) -> None:
        rwg = RandomWalkGenerator(self.X_all.tocsr())
        self.W = rwg.run_with_restart(
            self.decay, self.cutoff, self.n_samples, self.n_thread, self.random_seed
        )
        self.W = self.W.tocsc() / self.n_samples
