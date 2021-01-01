from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np

from irspack.user_cold_start.recommenders import base
from irspack.utils import get_n_threads

from ..evaluator import METRIC_NAMES, EvaluatorCore, Metrics
from ..recommenders.base import InteractionMatrix


class UserColdStartEvaluator:
    def __init__(
        self,
        X: InteractionMatrix,
        profiles: base.ProfileMatrix,
        mb_size: int = 1024,
        n_threads: Optional[int] = None,
        cutoff: int = 20,
    ):
        assert X.shape[0] == profiles.shape[0]

        self.core = EvaluatorCore(X.astype(np.float64), [])
        self.profiles = profiles
        self.n_users = X.shape[0]
        self.n_items = X.shape[1]
        self.dim_profile = profiles.shape[1]
        self.mb_size = mb_size
        self.n_threads = get_n_threads(n_threads)
        self.cutoff = cutoff

    def get_score(self, model: base.BaseUserColdStartRecommender) -> Dict[str, Any]:
        metric_base = Metrics(self.n_items)
        for start in range(0, self.n_users, self.mb_size):
            end = min(start + self.mb_size, self.n_users)
            score_mb = model.get_score(self.profiles[start:end])
            metric = self.core.get_metrics(
                score_mb, self.cutoff, start, self.n_threads, False
            )
            metric_base.merge(metric)
        return metric_base.as_dict()

    def get_scores(
        self, model: base.BaseUserColdStartRecommender, cutoffs: List[int]
    ) -> Dict[str, float]:
        result: Dict[str, float] = OrderedDict()
        scores = self.get_scores_as_list(model, cutoffs)
        for cutoff, score in zip(cutoffs, scores):
            for metric_name in METRIC_NAMES:
                result[f"{metric_name}@{cutoff}"] = score[metric_name]
        return result

    def get_scores_as_list(
        self, model: base.BaseUserColdStartRecommender, cutoffs: List[int]
    ) -> List[Dict[str, float]]:
        n_items = model.n_items
        metrics: List[Metrics] = []
        for c in cutoffs:
            metrics.append(Metrics(n_items))
        n_validated = self.n_users
        block_end = n_validated
        mb_size = self.mb_size

        for chunk_start in range(0, block_end, mb_size):
            chunk_end = min(chunk_start + mb_size, block_end)
            score_mb = model.get_score(self.profiles[chunk_start:chunk_end])
            for i, cutoff in enumerate(cutoffs):
                chunked_metric = self.core.get_metrics(
                    score_mb, cutoff, chunk_start, self.n_threads, False
                )
                metrics[i].merge(chunked_metric)
        return [item.as_dict() for item in metrics]
