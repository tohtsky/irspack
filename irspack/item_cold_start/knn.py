from ..definitions import UserIndexArray
from ..recommenders.base import DenseScoreArray
import numpy as np
from scipy import sparse as sps
from typing import Optional

from .base import ItemColdStartRecommenderBase, ProfileMatrix, InteractionMatrix
from ..recommenders._knn import CosineKNNComputer
from ..parameter_tuning import IntegerSuggestion, LogUniformSuggestion


class ItemCBKNNRecommender(ItemColdStartRecommenderBase):
    suggest_param_range = [
        IntegerSuggestion("top_k", 5, 2000),
        LogUniformSuggestion("shrink", 1e-2, 1e2),
    ]
    sim_computer: Optional[CosineKNNComputer]

    def __init__(
        self,
        X_interaction: InteractionMatrix,
        X_profile: ProfileMatrix,
        top_k: int = 100,
        n_thread: Optional[int] = 1,
        shrink: float = 1e-1,
    ):
        if n_thread is None:
            n_thread = 1
        assert n_thread >= 1
        self.top_k = top_k
        self.shrink = shrink
        self.n_thread = n_thread
        super().__init__(X_interaction, X_profile)
        self.X_interaction_csc = X_interaction.astype(np.float64).tocsc()
        self.sim_computer = None

    def _learn(self) -> None:
        self.sim_computer = CosineKNNComputer(
            self.X_profile, self.n_thread, self.shrink
        )

    def get_score_for_user_range(
        self, user_range: UserIndexArray, item_profile: ProfileMatrix
    ) -> DenseScoreArray:
        if self.sim_computer is None:
            raise RuntimeError("'get_score' called before learn.")
        if not sps.issparse(item_profile):
            item_profile = sps.csr_matrix(item_profile)
        similarity = self.sim_computer.compute_block(item_profile, self.top_k)
        score = self.X_interaction[user_range].dot(similarity.T)
        return score.astype(np.float64).toarray()
