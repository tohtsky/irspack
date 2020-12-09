"""
Copyright 2020 BizReach, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from enum import Enum

import numpy as np
from typing import List, Dict
from collections import OrderedDict

from .recommenders import base as base_recommender
from ._evaluator import EvaluatorCore, Metrics
from .definitions import InteractionMatrix


class TargetMetric(Enum):
    NDCG = "ndcg"
    RECALL = "recall"
    HIT = "hit"


METRIC_NAMES = ["hit", "recall", "ndcg", "gini_index", "entropy", "appeared_item"]


class Evaluator(object):
    def __init__(
        self,
        ground_truth: InteractionMatrix,
        offset: int,
        cutoff: int = 10,
        target_metric: str = "ndcg",
        n_thread: int = 1,
        mb_size: int = 1024,
    ):
        ground_truth = ground_truth.tocsr().astype(np.float64)
        ground_truth.sort_indices()
        self.core = EvaluatorCore(ground_truth)
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
        n_item = model.n_item
        metrics: List[Metrics] = []
        for c in cutoffs:
            metrics.append(Metrics(n_item))

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
        n_thread: int = 1,
        mb_size: int = 1024,
    ):
        super(EvaluatorWithColdUser, self).__init__(
            ground_truth, 0, cutoff, target_metric, n_thread, mb_size
        )
        self.input_interaction = input_interaction

    def get_scores_as_list(
        self, model: "base_recommender.BaseRecommender", cutoffs: List[int]
    ) -> List[Dict[str, float]]:
        if not isinstance(
            model, base_recommender.BaseRecommenderWithColdStartPredictability
        ):
            raise ValueError(
                "The Recommender must be a subtype of BaseRecommenderWithColdStartPredictability. "
            )

        n_item = model.n_item
        metrics: List[Metrics] = []
        for c in cutoffs:
            metrics.append(Metrics(n_item))

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
