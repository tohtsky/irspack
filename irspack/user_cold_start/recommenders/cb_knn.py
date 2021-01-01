from typing import Optional

import numpy as np
from scipy import sparse as sps

from irspack.definitions import DenseScoreArray
from irspack.recommenders._knn import CosineSimilarityComputer
from irspack.utils import get_n_threads, sparse_mm_threaded

from .base import BaseUserColdStartRecommender, InteractionMatrix, ProfileMatrix


class UserCBCosineKNNRecommender(BaseUserColdStartRecommender):
    def __init__(
        self,
        X_interaction: InteractionMatrix,
        X_profile: ProfileMatrix,
        top_k: int = 100,
        n_threads: Optional[int] = None,
        shrink: float = 1e-1,
    ):

        self.top_k = top_k
        self.shrink = shrink
        self.n_threads = get_n_threads(n_threads)
        super().__init__(X_interaction, X_profile)
        self.X_interaction_csc = X_interaction.astype(np.float64).tocsc()
        self.sim_computer: Optional[CosineSimilarityComputer] = None

    def _learn(self) -> None:
        self.sim_computer = CosineSimilarityComputer(
            self.X_profile, self.shrink, normalize=True, n_threads=self.n_threads
        )

    def get_score(self, profile: ProfileMatrix) -> DenseScoreArray:
        if self.sim_computer is None:
            raise RuntimeError("'get_score' called before learn.")
        if not sps.issparse(profile):
            profile = sps.csr_matrix(profile)
        similarity = self.sim_computer.compute_similarity(profile, self.top_k)
        score = sparse_mm_threaded(similarity, self.X_interaction_csc, self.n_threads)
        return score.astype(np.float64).toarray()
