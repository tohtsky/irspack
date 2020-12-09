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
import gc

import numpy as np
from scipy import linalg

from ..definitions import InteractionMatrix
from ..parameter_tuning import LogUniformSuggestion
from .base import BaseSimilarityRecommender


class DenseSLIMRecommender(BaseSimilarityRecommender):
    default_tune_range = [LogUniformSuggestion("reg", 1, 1e4)]

    def __init__(self, X_all: InteractionMatrix, reg: float = 1):
        super(DenseSLIMRecommender, self).__init__(X_all)
        self.reg = reg

    def learn(self) -> None:
        X_all_f32 = self.X_all.astype(np.float32)
        P = X_all_f32.T.dot(X_all_f32)
        P_dense: np.ndarray = P.todense()
        del P
        P_dense[np.arange(self.n_item), np.arange(self.n_item)] += self.reg
        gc.collect()
        P_dense = linalg.inv(P_dense, overwrite_a=True)

        gc.collect()
        diag_P_inv = 1 / np.diag(P_dense)
        P_dense *= -diag_P_inv[np.newaxis, :]
        range_ = np.arange(self.n_item)
        P_dense[range_, range_] = 0
        self.W = P_dense
