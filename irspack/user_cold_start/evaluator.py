import numpy as np
from typing import Dict, Any, List
from collections import OrderedDict
from ..evaluator import METRIC_NAMES, EvaluatorCore, Metrics
from ..recommenders.base import InteractionMatrix

from . import base


class UserColdStartEvaluator:
    def __init__(
        self,
        X: InteractionMatrix,
        profiles: base.ProfileMatrix,
        mb_size: int = 1024,
        n_thread: int = 1,
    ):
        assert X.shape[0] == profiles.shape[0]

        self.core = EvaluatorCore(X.astype(np.float64))
        self.profiles = profiles
        self.n_user = X.shape[0]
        self.n_item = X.shape[1]
        self.dim_profile = profiles.shape[1]
        self.mb_size = mb_size
        self.n_thread = n_thread

    def get_score(
        self, model: base.BaseUserColdStartRecommender, cutoff: int = 20
    ) -> Dict[str, Any]:
        metric_base = Metrics(self.n_item)
        for start in range(0, self.n_user, self.mb_size):
            end = min(start + self.mb_size, self.n_user)
            score_mb = model.get_score(self.profiles[start:end])
            metric = self.core.get_metrics(
                score_mb, cutoff, start, self.n_thread, False
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
        n_item = model.n_item
        metrics: List[Metrics] = []
        for c in cutoffs:
            metrics.append(Metrics(n_item))
        n_validated = self.n_user
        block_end = n_validated
        mb_size = self.mb_size

        for chunk_start in range(0, block_end, mb_size):
            chunk_end = min(chunk_start + mb_size, block_end)
            score_mb = model.get_score(self.profiles[chunk_start:chunk_end])
            for i, cutoff in enumerate(cutoffs):
                chunked_metric = self.core.get_metrics(
                    score_mb, cutoff, chunk_start, self.n_thread, False
                )
                metrics[i].merge(chunked_metric)
        return [item.as_dict() for item in metrics]
