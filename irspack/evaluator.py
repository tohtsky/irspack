import warnings
from collections import OrderedDict
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from irspack._evaluator import EvaluatorCore, Metrics
from irspack.definitions import InteractionMatrix
from irspack.utils import get_n_threads

if TYPE_CHECKING:
    from .recommenders import base as base_recommender


class TargetMetric(Enum):
    NDCG = "ndcg"
    RECALL = "recall"
    HIT = "hit"
    MAP = "map"
    PRECISION = "precision"


METRIC_NAMES = [
    "hit",
    "recall",
    "ndcg",
    "map",
    "precision",
    "gini_index",
    "entropy",
    "appeared_item",
]


class Evaluator:
    """Evaluates recommenders' performance against validation set.

    Args:
        ground_truth (Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
            The held-out ground-truth.
        offset (int):
            Where the validation target user block begins.
            Often the validation set is defined for a subset of users.
            When offset is not 0, we assume that the users with validation
            ground truth corresponds to X_train[offset:] where X_train
            is the matrix feeded into the recommender class.
        cutoff (int, optional):
            Controls the default number of recommendation.
            When the evaluator is used for parameter tuning, this cutoff value will be used.
            Defaults to 10.
        target_metric (str, optional):
            Specifies the target metric when this evaluator is used for
            parameter tuning. Defaults to "ndcg".
        recommendable_items (Optional[List[int]], optional):
            Global recommendable items. Defaults to None.
            If this parameter is not None, evaluator will be concentrating on
            the recommender's score output for these recommendable_items,
            and compute the ranking performance within this subset.
        per_user_recommendable_items (Optional[List[List[int]]], optional):
            Similar to `recommendable_items`, but this time the recommendable items can vary among users. Defaults to None.
        n_threads (int, optional):
            Specifies the Number of threads to sort scores and compute the evaluation metrics.
            If ``None``, the environment variable ``"IRSPACK_NUM_THREADS_DEFAULT"`` will be looked up,
            and if there is no such an environment variable, it will be set to 1. Defaults to None.
        mb_size (int, optional):
            The rows of chunked user score. Defaults to 1024.
    """

    n_users: int
    n_items: int

    def __init__(
        self,
        ground_truth: InteractionMatrix,
        offset: int = 0,
        cutoff: int = 10,
        target_metric: str = "ndcg",
        recommendable_items: Optional[List[int]] = None,
        per_user_recommendable_items: Optional[List[List[int]]] = None,
        n_threads: Optional[int] = None,
        mb_size: int = 1024,
    ) -> None:

        ground_truth = ground_truth.tocsr().astype(np.float64)
        ground_truth.sort_indices()
        if recommendable_items is None:
            if per_user_recommendable_items is None:
                recommendable_items_arg: List[List[int]] = []
            else:
                recommendable_items_arg = per_user_recommendable_items
        else:
            recommendable_items_arg = [recommendable_items]

        self.core = EvaluatorCore(ground_truth, recommendable_items_arg)
        self.offset = offset
        self.n_users = ground_truth.shape[0]
        self.n_items = ground_truth.shape[1]
        self.target_metric = TargetMetric(target_metric)
        self.cutoff = cutoff
        self.n_threads = get_n_threads(n_threads)
        self.mb_size = mb_size

    def get_target_score(self, model: "base_recommender.BaseRecommender") -> float:
        """Compute the optimization target score (self.target_metric) with the cutoff being ``self.cutoff``.

        Args:
            model: The evaluated model.

        Returns:
            The metric value.
        """
        return self.get_score(model)[self.target_metric.value]

    def get_score(self, model: "base_recommender.BaseRecommender") -> Dict[str, float]:
        """Compute the score with the cutoff being ``self.cutoff``.

        Args:
            model : The evaluated recommender.

        Returns:
            metric values.
        """
        return self._get_scores_as_list(model, [self.cutoff])[0]

    def get_scores(
        self, model: "base_recommender.BaseRecommender", cutoffs: List[int]
    ) -> Dict[str, float]:
        """Compute the score with the specified cutoffs.

        Args:
            model : The evaluated recommender.
            cutoffs : for each value in cutoff, the class computes
                the metric values.

        Returns:
            The Resulting metric values. This time, the result
            will look like ``{"ndcg@20": 0.35, "map@20": 0.2, ...}``.
        """

        result: Dict[str, float] = OrderedDict()
        scores = self._get_scores_as_list(model, cutoffs)
        for cutoff, score in zip(cutoffs, scores):
            for metric_name in METRIC_NAMES:
                result[f"{metric_name}@{cutoff}"] = score[metric_name]
        return result

    def _get_scores_as_list(
        self, model: "base_recommender.BaseRecommender", cutoffs: List[int]
    ) -> List[Dict[str, float]]:
        if self.offset + self.n_users > model.n_users:
            raise ValueError("evaluator offset + n_users exceeds the model's n_users.")
        if self.n_items != model.n_items:
            raise ValueError("The model and evaluator assume different n_items.")
        n_items = self.n_items
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
                    scores, c, chunk_start - self.offset, self.n_threads, False
                )
                metrics[i].merge(chunked_metric)

        return [item.as_dict() for item in metrics]


class EvaluatorWithColdUser(Evaluator):
    """Evaluates recommenders' performance against cold (unseen) users.

    Args:
        input_interaction (Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]): The cold-users' known interaction
            with the items.
        ground_truth (Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]): The held-out ground-truth.
        offset (int): Where the validation target user block begins.
            Often the validation set is defined for a subset of users.
            When offset is not 0, we assume that the users with validation
            ground truth corresponds to X_train[offset:] where X_train
            is the matrix feeded into the recommender class.
        cutoff (int, optional):
            Controls the number of recommendation.
            Defaults to 10.
        target_metric (str, optional):
            Optimization target metric.
            Defaults to "ndcg".
        recommendable_items (Optional[List[int]], optional):
            Global recommendable items. Defaults to None.
            If this parameter is not None, evaluator will be concentrating on
            the recommender's score output for these recommendable_items,
            and compute the ranking performance within this subset.
        per_user_recommendable_items (Optional[List[List[int]]], optional):
            Similar to `recommendable_items`, but this time the recommendable items can vary among users. Defaults to None.
        n_threads (int, optional):
            Specifies the Number of threads to sort scores and compute the evaluation metrics.
            If ``None``, the environment variable ``"IRSPACK_NUM_THREADS_DEFAULT"`` will be looked up,
            and if there is no such an environment variable, it will be set to 1. Defaults to None.
        mb_size (int, optional):
            The rows of chunked user score. Defaults to 1024.
    """

    def __init__(
        self,
        input_interaction: InteractionMatrix,
        ground_truth: InteractionMatrix,
        cutoff: int = 10,
        target_metric: str = "ndcg",
        recommendable_items: Optional[List[int]] = None,
        per_item_recommendable_items: Optional[List[List[int]]] = None,
        n_threads: Optional[int] = None,
        mb_size: int = 1024,
    ):

        super(EvaluatorWithColdUser, self).__init__(
            ground_truth,
            0,
            cutoff,
            target_metric,
            recommendable_items,
            per_item_recommendable_items,
            n_threads,
            mb_size,
        )
        self.input_interaction = input_interaction

    def _get_scores_as_list(
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
            if not scores.flags.c_contiguous:
                warnings.warn(
                    "Found col-major(fortran-style) score values.\n"
                    "Transforming it to row-major score matrix."
                )
                scores = np.ascontiguousarray(scores, dtype=np.float64)

            for i, c in enumerate(cutoffs):
                chunked_metric = self.core.get_metrics(
                    scores, c, chunk_start, self.n_threads, False
                )
                metrics[i].merge(chunked_metric)

        return [item.as_dict() for item in metrics]
