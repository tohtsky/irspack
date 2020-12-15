from .base import BaseSimilarityRecommender, BaseRecommenderWithThreadingSupport
from ..definitions import InteractionMatrix
from ..parameter_tuning import (
    LogUniformSuggestion,
    IntegerSuggestion,
    CategoricalSuggestion,
)
from typing import Optional
from ._knn import P3alphaComputer


class P3alphaRecommender(
    BaseSimilarityRecommender, BaseRecommenderWithThreadingSupport
):
    default_tune_range = [
        LogUniformSuggestion("alpha", low=1e-10, high=2),
        IntegerSuggestion("top_k", low=10, high=1000),
        CategoricalSuggestion("normalize_weight", [True, False]),
    ]

    def __init__(
        self,
        X_all: InteractionMatrix,
        alpha: float = 1,
        top_k: Optional[int] = None,
        normalize_weight: bool = False,
        n_thread: Optional[int] = 1,
    ):
        super().__init__(X_all, n_thread=n_thread)
        self.alpha = alpha
        self.top_k = top_k
        self.normalize_weight = normalize_weight

    def learn(self):
        computer = P3alphaComputer(
            self.X_all.T,
            alpha=self.alpha,
            normalize=self.normalize_weight,
            n_thread=self.n_thread,
        )
        top_k = self.X_all.shape[1] if self.top_k is None else self.top_k
        self.W = computer.compute_W(self.X_all.T, top_k)
