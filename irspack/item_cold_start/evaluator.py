import numpy as np
from scipy import sparse as sps
from typing import Dict, Any
from .._evaluator import EvaluatorCore, Metrics
from ..recommenders.base import InteractionMatrix
from . import base


class ItemColdStartEvaluator(object):
    def __init__(
        self, X: InteractionMatrix, profiles: "base.ProfileMatrix", mb_size: int = 1024
    ):
        X = sps.csr_matrix(X)
        X.sort_indices()
        assert X.shape[1] == profiles.shape[0]

        self.core = EvaluatorCore(X.astype(np.float64))
        self.profiles = profiles
        self.n_users: int = X.shape[0]
        self.n_items: int = X.shape[1]
        self.dim_profile = profiles.shape[1]
        self.mb_size = mb_size

    def get_score(
        self, model: "base.ItemColdStartRecommenderBase", cutoff=20
    ) -> Dict[str, Any]:
        metric_base = Metrics(self.n_items)
        for start in range(0, self.n_users, self.mb_size):
            end = min(start + 1024, self.n_users)
            score_mb = model.get_score_for_user_range(
                np.arange(start, end), self.profiles
            )
            metric = self.core.get_metrics(score_mb, cutoff, start, 4, False)
            metric_base.merge(metric)
        return metric_base.as_dict()
