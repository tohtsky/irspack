import numpy as np
from .base import UserColdStartRecommenderBase, ProfileMatrix


class TopPopularRecommender(UserColdStartRecommenderBase):
    def learn(self):
        self.popularity = self.X_interaction.sum(axis=0).astype(np.float64)

    def get_score(self, profile: ProfileMatrix):
        n_users = profile.shape[0]
        return np.repeat(self.popularity, n_users, axis=0)
