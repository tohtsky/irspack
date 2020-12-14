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
