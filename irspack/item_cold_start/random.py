from ..definitions import UserIndexArray
from ..recommenders.base import DenseScoreArray
import numpy as np

from .base import ItemColdStartRecommenderBase, ProfileMatrix, InteractionMatrix


class RandomRecommender(ItemColdStartRecommenderBase):
    def __init__(
        self,
        X_interaction: InteractionMatrix,
        X_profile: ProfileMatrix,
    ):
        super().__init__(X_interaction, X_profile)

    def _learn(self) -> None:
        pass

    def get_score_for_user_range(
        self, user_index_range: UserIndexArray, item_profile: ProfileMatrix
    ) -> DenseScoreArray:
        return np.random.rand(user_index_range.shape[0], item_profile.shape[0])
