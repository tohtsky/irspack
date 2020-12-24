from typing import List

from irspack.parameter_tuning import (
    CategoricalSuggestion,
    IntegerSuggestion,
    LogUniformSuggestion,
    Suggestion,
)
from irspack.user_cold_start.recommenders.cb_knn import UserCBCosineKNNRecommender
from irspack.user_cold_start.recommenders.linear import LinearMethodRecommender
from irspack.user_cold_start.recommenders.popular import TopPopularRecommender

from .base_optimizer import BaseOptimizer


class TopPopularOptimizer(BaseOptimizer):
    recommender_class = TopPopularRecommender
    default_tune_range: List[Suggestion] = []


class LinearMethodOptimizer(BaseOptimizer):
    recommender_class = LinearMethodRecommender
    default_tune_range = [
        LogUniformSuggestion("reg", 1e-1, 1e4),
        CategoricalSuggestion("fit_intercept", [True, False]),
    ]


class UserCBCosineKNNOptimizer(BaseOptimizer):
    recommender_class = UserCBCosineKNNRecommender
    default_tune_range = [
        IntegerSuggestion("top_k", 5, 2000),
        LogUniformSuggestion("shrink", 1e-2, 1e2),
    ]


__all__ = ["TopPopularOptimizer", "LinearMethodOptimizer", "UserCBCosineKNNOptimizer"]
