import numpy as np
from scipy import sparse as sps
from typing import Optional


from irspack.definitions import DenseScoreArray
from .base import BaseUserColdStartRecommender, ProfileMatrix, InteractionMatrix
from irspack.recommenders._knn import CosineSimilarityComputer
from irspack.utils._util_cpp import sparse_mm_threaded
from irspack.parameter_tuning import IntegerSuggestion, LogUniformSuggestion


class UserCBCosineKNNRecommender(BaseUserColdStartRecommender):
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
        self.sim_computer: Optional[CosineSimilarityComputer] = None

    def _learn(self) -> None:
        self.sim_computer = CosineSimilarityComputer(
            self.X_profile, self.shrink, normalize=True, n_thread=self.n_thread
        )

    def get_score(self, profile: ProfileMatrix) -> DenseScoreArray:
        if self.sim_computer is None:
            raise RuntimeError("'get_score' called before learn.")
        if not sps.issparse(profile):
            profile = sps.csr_matrix(profile)
        similarity = self.sim_computer.compute_similarity(profile, self.top_k)
        score = sparse_mm_threaded(similarity, self.X_interaction_csc, self.n_thread)
        return score.astype(np.float64).toarray()
