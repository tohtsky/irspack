import numpy as np
import scipy.sparse as sps

from irspack.recommenders.base import BaseRecommender


class MockRecommender(BaseRecommender, register_class=False):
    def __init__(self, X_all: sps.csr_matrix, scores: np.ndarray) -> None:
        super().__init__(X_all)
        assert X_all.shape == scores.shape
        self.scores = scores

    def get_score(self, user_indices: np.ndarray) -> np.ndarray:
        scores: np.ndarray = self.scores[user_indices]
        return scores

    def _learn(self) -> None:
        pass
