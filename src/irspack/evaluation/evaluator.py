import warnings
from collections import OrderedDict
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Union

import numpy as np
from scipy import sparse as sps

from .._threading import get_n_threads
from ..definitions import DenseScoreArray, InteractionMatrix
from ._core_evaluator import EvaluatorCore, Metrics

if TYPE_CHECKING:
    from ..recommenders.base import BaseRecommender


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
        per_user_recommendable_items:
            Similar to `recommendable_items`, but this time the recommendable items can vary among users.
            If a sparse matrix is given, its nonzero indices are regarded as the list of recommendable items.
            Defaults to `None`.
        masked_interactions:
            If set, this matrix masks the score output of recommender model where it is non-zero.
            If none, the mask will be the training matrix itself owned by the recommender.

        n_threads:
            Specifies the Number of threads to sort scores and compute the evaluation metrics.
            If `None`, the environment variable ``"IRSPACK_NUM_THREADS_DEFAULT"` will be looked up,
            and if the variable is not set, it will be set to ``os.cpu_count()``. Defaults to `None`.

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
        per_user_recommendable_items: Union[
            None, List[List[int]], InteractionMatrix
        ] = None,
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
                if sps.issparse(per_user_recommendable_items):
                    per_user_as_csr = sps.csr_matrix(per_user_recommendable_items)
                    recommendable_items_arg = [
                        [int(j) for j in row.nonzero()[1]] for row in per_user_as_csr
                    ]
                else:
                    recommendable_items_arg = per_user_recommendable_items
                if len(recommendable_items_arg) != ground_truth.shape[0]:
                    raise ValueError(
                        "ground_truth and per_user_recommendable_items have inconsistent shapes."
                    )
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

    def get_target_score(self, model: "BaseRecommender") -> float:
        r"""Compute the optimization target score (self.target_metric) with the cutoff being ``self.cutoff``.

        Args:
            model: The evaluated model.

        Returns:
            The metric value.
        """
        return self.get_score(model)[self.target_metric.name]

    def get_score(self, model: "BaseRecommender") -> Dict[str, float]:
        r"""Compute the score with the cutoff being ``self.cutoff``.

        Args:
            model : The evaluated recommender.

        Returns:
            metric values.
        """
        return self._get_scores_as_list(model, [self.cutoff])[0]

    def get_scores(
        self, model: "BaseRecommender", cutoffs: List[int]
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

    def get_score_from_score_matrix(self, scores: DenseScoreArray) -> Dict[str, float]:
        r"""Compute metrics from a user-by-item score matrix.

        The score matrix must have shape ``(n_users, n_items)``, in the same
        order as the ground truth supplied to this evaluator. Scores are copied
        before masking, so the input array is not modified.

        When ``masked_interactions`` was supplied to the evaluator, its entries
        are excluded from the ranking. Otherwise, no entries are masked.

        Args:
            scores: A float32 or float64 user-by-item score matrix.

        Returns:
            Metric values for this evaluator's default cutoff.
        """
        return self._get_scores_from_score_matrix_as_list(scores, [self.cutoff])[0]

    def get_scores_from_score_matrix(
        self, scores: DenseScoreArray, cutoffs: List[int]
    ) -> Dict[str, float]:
        r"""Compute metrics at multiple cutoffs from a user-by-item score matrix.

        The score matrix must have shape ``(n_users, n_items)``, in the same
        order as the ground truth supplied to this evaluator. Scores are copied
        before masking, so the input array is not modified.

        Args:
            scores: A float32 or float64 user-by-item score matrix.
            cutoffs: Cutoffs at which to compute metrics.

        Returns:
            Metric values keyed by metric name and cutoff.
        """
        result: Dict[str, float] = OrderedDict()
        scores_as_list = self._get_scores_from_score_matrix_as_list(scores, cutoffs)
        for cutoff, score in zip(cutoffs, scores_as_list):
            for metric_name in METRIC_NAMES:
                result[f"{metric_name}@{cutoff}"] = score[metric_name]
        return result

    def get_score_from_score_chunks(
        self, score_chunks: Iterable[DenseScoreArray]
    ) -> Dict[str, float]:
        r"""Compute metrics from an iterable of consecutive score row-blocks.

        This is the streaming counterpart of :meth:`get_score_from_score_matrix`.
        Each yielded array must have shape ``(n_block_users, n_items)`` and the
        blocks must be supplied in the same row order as ``ground_truth``; the
        concatenated rows must add up to :attr:`n_users`. This avoids
        materializing the full ``(n_users, n_items)`` score matrix: a caller can
        stream blocks from a model and discard each block once it has been scored.

        Each block is copied before any mask is applied, so caller-owned arrays
        are never modified.

        Args:
            score_chunks: An iterable yielding user-by-item score blocks in row
                order. Blocks may have any positive number of rows; their dtype
                must be float32 or float64.

        Returns:
            Metric values for this evaluator's default cutoff.
        """
        return self._get_scores_from_score_chunks_as_list(score_chunks, [self.cutoff])[
            0
        ]

    def get_scores_from_score_chunks(
        self,
        score_chunks: Iterable[DenseScoreArray],
        cutoffs: List[int],
    ) -> Dict[str, float]:
        r"""Compute metrics at multiple cutoffs from consecutive score row-blocks.

        This is the streaming counterpart of :meth:`get_scores_from_score_matrix`.
        See :meth:`get_score_from_score_chunks` for the contract on
        ``score_chunks``.

        Args:
            score_chunks: An iterable yielding user-by-item score blocks in row
                order. Blocks may have any positive number of rows; their dtype
                must be float32 or float64.
            cutoffs: Cutoffs at which to compute metrics.

        Returns:
            Metric values keyed by metric name and cutoff.
        """
        result: Dict[str, float] = OrderedDict()
        scores_as_list = self._get_scores_from_score_chunks_as_list(
            score_chunks, cutoffs
        )
        for cutoff, score in zip(cutoffs, scores_as_list):
            for metric_name in METRIC_NAMES:
                result[f"{metric_name}@{cutoff}"] = score[metric_name]
        return result

    def _get_score_matrix_mask(self) -> Optional[sps.csr_matrix]:
        return self.masked_interactions

    def _get_scores_from_score_matrix_as_list(
        self, scores: DenseScoreArray, cutoffs: List[int]
    ) -> List[Dict[str, float]]:
        if scores.ndim != 2 or scores.shape != (self.n_users, self.n_items):
            raise ValueError(
                "score matrix must have shape "
                f"({self.n_users}, {self.n_items}), but got {scores.shape}."
            )
        if scores.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("score matrix must have dtype float32 or float64.")

        mb_size = self.mb_size
        chunks = (
            scores[chunk_start : chunk_start + mb_size]
            for chunk_start in range(0, self.n_users, mb_size)
        )
        return self._get_scores_from_score_chunks_as_list(chunks, cutoffs)

    def _get_scores_from_score_chunks_as_list(
        self,
        score_chunks: Iterable[DenseScoreArray],
        cutoffs: List[int],
    ) -> List[Dict[str, float]]:
        mask = self._get_score_matrix_mask()
        metrics = [Metrics(self.n_items) for _ in cutoffs]
        chunk_start = 0
        for score_chunk in score_chunks:
            if not isinstance(score_chunk, np.ndarray) or score_chunk.ndim != 2:
                raise ValueError(
                    "each score chunk must be a 2-D ndarray, got "
                    f"{type(score_chunk).__name__}."
                )
            if score_chunk.shape[1] != self.n_items:
                raise ValueError(
                    "score chunk must have n_items="
                    f"{self.n_items} columns, got {score_chunk.shape[1]}."
                )
            if score_chunk.dtype not in (np.dtype("float32"), np.dtype("float64")):
                raise ValueError("score chunk must have dtype float32 or float64.")
            chunk_end = chunk_start + score_chunk.shape[0]
            if chunk_end > self.n_users:
                raise ValueError(
                    "score chunks supplied more rows than the evaluator's "
                    f"n_users={self.n_users}: processed {chunk_end} rows."
                )
            if score_chunk.shape[0] == 0:
                continue
            # Masking scores must not alter an array owned by the caller.
            score_chunk = score_chunk.copy(order="C")
            if mask is not None:
                score_chunk[mask[chunk_start:chunk_end].nonzero()] = -np.inf
            for i, cutoff in enumerate(cutoffs):
                metrics[i].merge(self._get_metrics(score_chunk, cutoff, chunk_start))
            chunk_start = chunk_end
        if chunk_start != self.n_users:
            raise ValueError(
                "score chunks did not cover the evaluator's "
                f"n_users={self.n_users} rows: processed {chunk_start} rows."
            )
        return [item.as_dict() for item in metrics]

    def _get_scores_as_list(
        self, model: "BaseRecommender", cutoffs: List[int]
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
        per_user_recommendable_items: Union[
            None, List[List[int]], InteractionMatrix
        ] = None,
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

    def _get_score_matrix_mask(self) -> Optional[sps.csr_matrix]:
        if self.masked_interactions is None:
            return self.input_interaction.tocsr()
        return self.masked_interactions

    def _get_scores_as_list(
        self,
        model: "BaseRecommender",
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
