from ..definitions import DenseScoreArray
import numpy as np
from .base import BaseUserColdStartRecommender, ProfileMatrix


class TopPopularRecommender(BaseUserColdStartRecommender):
    def _learn(self) -> None:
        self.popularity = self.X_interaction.sum(axis=0).astype(np.float64)

    def get_score(self, profile: ProfileMatrix) -> DenseScoreArray:
        target_n_user = profile.shape[0]
        return np.repeat(self.popularity, target_n_user, axis=0)
