from collections import OrderedDict
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from ._evaluator import EvaluatorCore, Metrics
from .definitions import InteractionMatrix

if TYPE_CHECKING:
    from .recommenders import base as base_recommender


class TargetMetric(Enum):
    NDCG = "ndcg"
    RECALL = "recall"
    HIT = "hit"


METRIC_NAMES = [
    "hit",
    "recall",
    "ndcg",
    "map",
    "gini_index",
    "entropy",
    "appeared_item",
]


class Evaluator(object):
    def __init__(
        self,
        ground_truth: InteractionMatrix,
        offset: int,
        cutoff: int = 10,
        target_metric: str = "ndcg",
        recommendable_items: Optional[List[int]] = None,
        per_item_recommendable_items: Optional[List[List[int]]] = None,
        n_thread: int = 1,
        mb_size: int = 1024,
    ):
        ground_truth = ground_truth.tocsr().astype(np.float64)
        ground_truth.sort_indices()
        if recommendable_items is None:
            if per_item_recommendable_items is None:
                recommendable_items_arg: List[List[int]] = []
            else:
                recommendable_items_arg = per_item_recommendable_items
        else:
            recommendable_items_arg = [recommendable_items]

        self.core = EvaluatorCore(ground_truth, recommendable_items_arg)
        self.offset = offset
        self.n_users = ground_truth.shape[0]
        self.target_metric = TargetMetric(target_metric)
        self.cutoff = cutoff
        self.n_thread = n_thread
        self.mb_size = mb_size

    def get_score(self, model: "base_recommender.BaseRecommender") -> Dict[str, float]:
        return self.get_scores_as_list(model, [self.cutoff])[0]

    def get_scores(
        self, model: "base_recommender.BaseRecommender", cutoffs: List[int]
    ) -> Dict[str, float]:
        result: Dict[str, float] = OrderedDict()
        scores = self.get_scores_as_list(model, cutoffs)
        for cutoff, score in zip(cutoffs, scores):
            for metric_name in METRIC_NAMES:
                result[f"{metric_name}@{cutoff}"] = score[metric_name]
        return result

    def get_scores_as_list(
        self, model: "base_recommender.BaseRecommender", cutoffs: List[int]
    ) -> List[Dict[str, float]]:
        n_items = model.n_items
        metrics: List[Metrics] = []
        for c in cutoffs:
            metrics.append(Metrics(n_items))

        block_start = self.offset
        n_validated = self.n_users
        block_end = block_start + n_validated
        mb_size = self.mb_size

        for chunk_start in range(block_start, block_end, mb_size):
            chunk_end = min(chunk_start + mb_size, block_end)
            try:
                # try faster method
                scores = model.get_score_remove_seen_block(chunk_start, chunk_end)
            except NotImplementedError:
                # block-by-block
                scores = model.get_score_remove_seen(np.arange(chunk_start, chunk_end))
            for i, c in enumerate(cutoffs):
                chunked_metric = self.core.get_metrics(
                    scores, c, chunk_start - self.offset, self.n_thread, False
                )
                metrics[i].merge(chunked_metric)

        return [item.as_dict() for item in metrics]


class EvaluatorWithColdUser(Evaluator):
    def __init__(
        self,
        input_interaction: InteractionMatrix,
        ground_truth: InteractionMatrix,
        cutoff: int = 10,
        target_metric: str = "ndcg",
        recommendable_items: Optional[List[int]] = None,
        per_item_recommendable_items: Optional[List[List[int]]] = None,
        n_thread: int = 1,
        mb_size: int = 1024,
    ):
        super(EvaluatorWithColdUser, self).__init__(
            ground_truth,
            0,
            cutoff,
            target_metric,
            recommendable_items,
            per_item_recommendable_items,
            n_thread,
            mb_size,
        )
        self.input_interaction = input_interaction

    def get_scores_as_list(
        self,
        model: "base_recommender.BaseRecommender",
        cutoffs: List[int],
    ) -> List[Dict[str, float]]:

        n_items = model.n_items
        metrics: List[Metrics] = []
        for c in cutoffs:
            metrics.append(Metrics(n_items))

        block_start = self.offset
        n_validated = self.n_users
        block_end = block_start + n_validated
        mb_size = self.mb_size

        for chunk_start in range(block_start, block_end, mb_size):
            chunk_end = min(chunk_start + mb_size, block_end)
            scores = model.get_score_cold_user_remove_seen(
                self.input_interaction[chunk_start:chunk_end]
            )
            for i, c in enumerate(cutoffs):
                chunked_metric = self.core.get_metrics(
                    scores, c, chunk_start, self.n_thread, False
                )
                metrics[i].merge(chunked_metric)

        return [item.as_dict() for item in metrics]
