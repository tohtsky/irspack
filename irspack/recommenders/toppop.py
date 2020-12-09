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
from typing import Optional

import numpy as np

from ..definitions import DenseScoreArray, InteractionMatrix, UserIndexArray
from .base import BaseRecommenderWithColdStartPredictability


class TopPopRecommender(BaseRecommenderWithColdStartPredictability):
    score: Optional[np.ndarray]

    def __init__(self, X_train: InteractionMatrix):
        super().__init__(X_train)
        self.score = None

    def learn(self):
        self.score = self.X_all.sum(axis=0).astype(np.float64)

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        if self.score is None:
            raise RuntimeError("'get_score' called before fit.")
        n_users: int = user_indices.shape[0]
        return np.repeat(self.score, n_users, axis=0)

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        if self.score is None:
            raise RuntimeError("'get_score_cold_user' called before fit.")
        n_users: int = X.shape[0]
        return np.repeat(self.score, n_users, axis=0)
