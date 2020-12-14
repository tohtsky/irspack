from ..recommenders.base import DenseScoreArray
import numpy as np
from scipy import sparse as sps
from typing import Optional

from .base import UserColdStartRecommenderBase, ProfileMatrix, InteractionMatrix
from ..recommenders._knn import CosineKNNComputer
from ..utils._util_cpp import sparse_mm_threaded
from ..parameter_tuning import IntegerSuggestion, LogUniformSuggestion


class UserCBKNNRecommender(UserColdStartRecommenderBase):
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

    def learn(self) -> None:
        self.sim_computer = CosineKNNComputer(
            self.X_profile, self.n_thread, self.shrink
        )

    def get_score(self, profile: ProfileMatrix) -> DenseScoreArray:
        if self.sim_computer is None:
            raise RuntimeError("'get_score' called before learn.")
        if not sps.issparse(profile):
            profile = sps.csr_matrix(profile)
        similarity = self.sim_computer.compute_block(profile, self.top_k)
        score = sparse_mm_threaded(similarity, self.X_interaction_csc, self.n_thread)
        return score.astype(np.float64).toarray()
