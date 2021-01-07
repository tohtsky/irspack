from typing import Optional

import numpy as np

from ..definitions import DenseScoreArray, InteractionMatrix, UserIndexArray
from .base import BaseRecommender


class TopPopRecommender(BaseRecommender):
    """A simple recommender system based on the popularity of the items in the
    training set (without any personalization).

    Args:
        X_train Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
            Input interaction matrix.
    """

    score_: Optional[np.ndarray]

    def __init__(self, X_train: InteractionMatrix):

        super().__init__(X_train)
        self.score_ = None

    def _learn(self) -> None:
        self.score_ = self.X_train_all.sum(axis=0).astype(np.float64)

    @property
    def score(self) -> np.ndarray:
        if self.score_ is None:
            raise RuntimeError("The method called before ``learn``.")
        return self.score_

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        n_users: int = user_indices.shape[0]
        return np.repeat(self.score, n_users, axis=0)

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        n_users: int = X.shape[0]
        return np.repeat(self.score, n_users, axis=0)
