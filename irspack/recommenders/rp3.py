from typing import List, Optional

import numpy as np
from scipy import sparse as sps
from sklearn.preprocessing import normalize

from ..definitions import InteractionMatrix
from ._knn import RP3betaComputer
from .base import BaseRecommenderWithThreadingSupport, BaseSimilarityRecommender


class RP3betaRecommender(
    BaseSimilarityRecommender, BaseRecommenderWithThreadingSupport
):
    def __init__(
        self,
        X_all: InteractionMatrix,
        alpha: float = 1,
        beta: float = 0.6,
        top_k: Optional[int] = None,
        normalize_weight: bool = False,
        n_thread: Optional[int] = None,
    ):
        super().__init__(X_all, n_thread=n_thread)
        self.alpha = alpha
        self.beta = beta
        self.top_k = top_k
        self.normalize_weight = normalize_weight

    def _learn(self) -> None:
        computer = RP3betaComputer(
            self.X_all.T,
            alpha=self.alpha,
            beta=self.beta,
            n_thread=self.n_thread,
        )
        top_k = self.X_all.shape[1] if self.top_k is None else self.top_k
        self.W_ = computer.compute_W(self.X_all.T, top_k)
        if self.normalize_weight:
            self.W_ = normalize(self.W_, norm="l1", axis=1)
