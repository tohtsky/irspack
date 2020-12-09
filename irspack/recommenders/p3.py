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
from .base import BaseSimilarityRecommender
from ..definitions import InteractionMatrix
from ..parameter_tuning import (
    LogUniformSuggestion,
    IntegerSuggestion,
    CategoricalSuggestion,
)
from typing import Optional, List
import numpy as np
from scipy import sparse as sps
from sklearn.preprocessing import normalize
from .utils import restrict_topk_columnwise


class P3alphaRecommender(BaseSimilarityRecommender):
    default_tune_range = [
        LogUniformSuggestion("alpha", low=1e-10, high=2),
        IntegerSuggestion("top_k", low=10, high=1000),
        CategoricalSuggestion("normalize_weight", [True, False]),
    ]

    def __init__(
        self,
        X_all: InteractionMatrix,
        alpha: float = 1,
        top_k: Optional[int] = None,
        normalize_weight: bool = False,
    ):
        super().__init__(X_all)
        self.alpha = alpha
        self.top_k = top_k
        self.normalize_weight = normalize_weight

    def learn(self):
        Pui = self.X_all.tocsc()
        Pui.data = np.power(Pui.data, self.alpha)
        Pui = normalize(Pui, norm="l1", axis=1)

        Piu = self.X_all.transpose().tocsr()
        Piu.data = np.power(Piu.data, self.alpha)
        Piu = normalize(Piu, norm="l1", axis=1)

        n_item = self.X_all.shape[1]

        # chunking
        MB_size = 1000
        Ws: List[sps.csr_matrix] = []
        for start in range(0, n_item, MB_size):
            end = min(n_item, start + MB_size)
            W_mb: sps.csc_matrix = Piu.dot(Pui[:, start:end]).tocsc()
            if self.top_k is not None:
                W_mb = restrict_topk_columnwise(W_mb, self.top_k)
            Ws.append(W_mb)
        self.W: sps.csc_matrix = sps.hstack(Ws, format="csc")
        self.W.eliminate_zeros()

        if self.normalize_weight:
            self.W = normalize(self.W, norm="l1", axis=1)

        self.W.sort_indices()
