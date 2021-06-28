import warnings
from collections import OrderedDict
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from scipy import sparse as sps

from irspack.definitions import DenseScoreArray, InteractionMatrix
from irspack.evaluator._core import EvaluatorCore, Metrics
from irspack.utils.threading import get_n_threads

if TYPE_CHECKING:
    from irspack.recommenders import base as base_recommender


class TargetMetric(Enum):
    ndcg = auto()
    recall = auto()
    hit = auto()
    map = auto()
    precision = auto()


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
    r"""Evaluates recommenders' performance against validation set.

    Args:
        ground_truth (Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
            The ground-truth.
        offset (int):
            Where the validation target user block begins.
            Often the validation set is defined for a subset of users.
            When offset is not 0, we assume that the users with validation
            ground truth corresponds to X_train[offset:] where X_train
            is the matrix feeded into the recommender class. Defaults to 0.
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
        masked_interactions (Optional[Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]], optional):
            If set, this matrix masks the score output of recommender model where it is non-zero.
            If none, the mask will be the training matrix itself owned by the recommender.

        n_threads (int, optional):
            Specifies the Number of threads to sort scores and compute the evaluation metrics.
            If ``None``, the environment variable ``"IRSPACK_NUM_THREADS_DEFAULT"`` will be looked up,
            and if the variable is not set, it will be set to ``os.cpu_count()``. Defaults to None.

        recall_with_cutoff (bool, optional):
            This affects the definition of recall.
            If ``True``, for each user, recall will be computed as

            .. math ::

                \frac{N_{\text{hit}}}{\min(\text{cutoff}, N_{\text{ground truth}})}

            If ``False``, this will be

            .. math ::

                \frac{N_{\text{hit}}}{N_{\text{ground truth}}}


        mb_size (int, optional):
            The rows of chunked user score. Defaults to 1024.
    """

    n_users: int
    n_items: int
    masked_interactions: Optional[sps.csr_matrix]

    def __init__(
        self,
        ground_truth: InteractionMatrix,
        offset: int = 0,
        cutoff: int = 10,
        target_metric: str = "ndcg",
        recommendable_items: Optional[List[int]] = None,
        per_user_recommendable_items: Optional[List[List[int]]] = None,
        masked_interactions: Optional[InteractionMatrix] = None,
        n_threads: Optional[int] = None,
        recall_with_cutoff: bool = False,
        mb_size: int = 128,
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
        self.target_metric = TargetMetric[target_metric]
        self.cutoff = cutoff
        self.target_metric_name = f"{self.target_metric.name}@{self.cutoff}"
        self.n_threads = get_n_threads(n_threads)
        self.mb_size = mb_size
        if masked_interactions is None:
            self.masked_interactions = None
        else:
            if masked_interactions.shape != ground_truth.shape:
                raise ValueError(
                    "ground_truth and masked_interactions have different shapes. "
                )
            self.masked_interactions = sps.csr_matrix(masked_interactions)

        self.recall_with_cutoff = recall_with_cutoff

    def _get_metrics(
        self, scores: DenseScoreArray, cutoff: int, ground_truth_begin: int
    ) -> Metrics:
        if scores.dtype == np.float64:
            return self.core.get_metrics_f64(
                scores,
                cutoff,
                ground_truth_begin,
                self.n_threads,
                self.recall_with_cutoff,
            )
        elif scores.dtype == np.float32:
            return self.core.get_metrics_f32(
                scores,
                cutoff,
                ground_truth_begin,
                self.n_threads,
                self.recall_with_cutoff,
            )
        else:
            raise ValueError("score must be either float32 or float64.")

    def get_target_score(self, model: "base_recommender.BaseRecommender") -> float:
        r"""Compute the optimization target score (self.target_metric) with the cutoff being ``self.cutoff``.

        Args:
            model: The evaluated model.

        Returns:
            The metric value.
        """
        return self.get_score(model)[self.target_metric.name]

    def get_score(self, model: "base_recommender.BaseRecommender") -> Dict[str, float]:
        r"""Compute the score with the cutoff being ``self.cutoff``.

        Args:
            model : The evaluated recommender.

        Returns:
            metric values.
        """
        return self._get_scores_as_list(model, [self.cutoff])[0]

    def get_scores(
        self, model: "base_recommender.BaseRecommender", cutoffs: List[int]
    ) -> Dict[str, float]:
        r"""Compute the score with the specified cutoffs.

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
                scores = model.get_score_block(chunk_start, chunk_end)
            except NotImplementedError:
                # block-by-block
                scores = model.get_score(np.arange(chunk_start, chunk_end))

            if self.masked_interactions is None:
                mask = model.X_train_all[chunk_start:chunk_end]
            else:
                mask = self.masked_interactions[
                    chunk_start - self.offset : chunk_end - self.offset
                ]
            scores[mask.nonzero()] = -np.inf
            for i, c in enumerate(cutoffs):
                chunked_metric = self._get_metrics(
                    scores,
                    c,
                    chunk_start - self.offset,
                )
                metrics[i].merge(chunked_metric)

        return [item.as_dict() for item in metrics]


class EvaluatorWithColdUser(Evaluator):
    r"""Evaluates recommenders' performance against cold (unseen) users.

    Args:
        input_interaction (Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
            The cold-users' known interaction with the items.
        ground_truth (Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
            The held-out ground-truth.
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
        masked_interactions (Optional[Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]], optional):
            If set, this matrix masks the score output of recommender model where it is non-zero.
            If none, the mask will be the training matrix (``input_interaction``) it self.
        n_threads (int, optional):
            Specifies the Number of threads to sort scores and compute the evaluation metrics.
            If ``None``, the environment variable ``"IRSPACK_NUM_THREADS_DEFAULT"`` will be looked up,
            and if the variable is not set, it will be set to ``os.cpu_count()``. Defaults to None.
        recall_with_cutoff (bool, optional):
            This affects the definition of recall.
            If ``True``, for each user, recall will be evaluated by

            .. math ::

                \frac{N_{\text{hit}}}{\min( \text{cutoff}, N_{\text{ground truth}} )}

            If ``False``, this will be

            .. math ::

                \frac{N_{\text{hit}}}{N_{\text{ground truth}}}

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
        per_user_recommendable_items: Optional[List[List[int]]] = None,
        masked_interactions: Optional[InteractionMatrix] = None,
        n_threads: Optional[int] = None,
        recall_with_cutoff: bool = False,
        mb_size: int = 1024,
    ):

        super().__init__(
            ground_truth,
            offset=0,
            cutoff=cutoff,
            target_metric=target_metric,
            recommendable_items=recommendable_items,
            per_user_recommendable_items=per_user_recommendable_items,
            masked_interactions=masked_interactions,
            n_threads=n_threads,
            recall_with_cutoff=recall_with_cutoff,
            mb_size=mb_size,
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
            scores = model.get_score_cold_user(
                self.input_interaction[chunk_start:chunk_end]
            )
            if self.masked_interactions is None:
                mask = self.input_interaction[chunk_start:chunk_end]
            else:
                mask = self.masked_interactions[chunk_start:chunk_end]
            scores[mask.nonzero()] = -np.inf

            if not scores.flags.c_contiguous:
                warnings.warn(
                    "Found col-major(fortran-style) score values.\n"
                    "Transforming it to row-major score matrix."
                )
                scores = np.ascontiguousarray(scores, dtype=np.float64)

            for i, c in enumerate(cutoffs):
                chunked_metric = self._get_metrics(scores, c, chunk_start)
                metrics[i].merge(chunked_metric)

        return [item.as_dict() for item in metrics]
