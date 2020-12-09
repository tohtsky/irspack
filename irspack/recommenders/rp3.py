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
from typing import List, Optional

import numpy as np
from scipy import sparse as sps
from sklearn.preprocessing import normalize

from ..definitions import InteractionMatrix
from ..parameter_tuning import (
    CategoricalSuggestion,
    IntegerSuggestion,
    LogUniformSuggestion,
)
from .base import BaseSimilarityRecommender
from .utils import restrict_topk_columnwise


class RP3betaRecommender(BaseSimilarityRecommender):
    default_tune_range = [
        LogUniformSuggestion("alpha", 1e-5, 10),
        IntegerSuggestion("top_k", 2, 4000),
        LogUniformSuggestion("beta", 1e-5, 5e-1),
        CategoricalSuggestion("normalize_weight", [True, False]),
    ]

    def __init__(
        self,
        X_all: InteractionMatrix,
        alpha: float = 1,
        beta: float = 0.6,
        top_k: Optional[int] = None,
        normalize_weight: bool = False,
    ):
        super().__init__(X_all)
        self.alpha = alpha
        self.beta = beta
        self.top_k = top_k
        self.normalize_weight = normalize_weight

    def learn(self) -> None:
        Pui = self.X_all.tocsc()
        Pui.data = np.power(Pui.data, self.alpha)
        Pui = normalize(Pui, norm="l1", axis=1)

        Piu = self.X_all.transpose().tocsr()
        Piu.data = np.power(Piu.data, self.alpha)
        Piu = normalize(Piu, norm="l1", axis=1)
        discount_factor = Piu.sum(axis=1).A1.astype(np.float64)
        mask_ = discount_factor != 0
        discount_factor[mask_] = np.power(discount_factor[mask_], -self.beta)
        n_item = self.X_all.shape[1]

        # chunking
        MB_size = 1000
        Ws: List[sps.csc_matrix] = []
        for start in range(0, n_item, MB_size):
            end = min(n_item, start + MB_size)
            W_mb = Piu.dot(Pui[:, start:end]).tocoo()
            W_mb.data *= discount_factor[start + W_mb.col]
            W_mb = W_mb.tocsc()
            if self.top_k is not None:
                W_mb = restrict_topk_columnwise(W_mb, self.top_k)
            Ws.append(W_mb)

        W_un_normalized: sps.csc_matrix = sps.hstack(Ws, format="csc")

        if self.normalize_weight:
            self.W = normalize(W_un_normalized, norm="l1", axis=1)
        else:
            self.W = W_un_normalized

        self.W.eliminate_zeros()
        self.W.sort_indices()
