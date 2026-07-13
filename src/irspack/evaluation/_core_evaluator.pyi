from collections.abc import Sequence
from typing import Annotated

import numpy
import scipy
from numpy.typing import NDArray

class Metrics:
    def __init__(self, n_item: int) -> None: ...
    def merge(self, other: Metrics) -> None: ...
    def as_dict(self) -> dict[str, float]: ...

class EvaluatorCore:
    def __init__(
        self,
        ground_truth: scipy.sparse.csr_matrix[float],
        recommendable: Sequence[Sequence[int]],
    ) -> None: ...
    def get_metrics_f64(
        self,
        score_array: Annotated[NDArray[numpy.float64], dict(shape=(None, None))],
        cutoff: int,
        offset: int,
        n_threads: int,
        recall_with_cutoff: bool = False,
    ) -> Metrics: ...
    def get_metrics_f32(
        self,
        score_array: Annotated[NDArray[numpy.float32], dict(shape=(None, None))],
        cutoff: int,
        offset: int,
        n_threads: int,
        recall_with_cutoff: bool = False,
    ) -> Metrics: ...
    def get_ground_truth(self) -> scipy.sparse.csr_matrix[float]: ...
    def cache_X_as_set(self, n_threads: int) -> None: ...
    def __getstate__(
        self,
    ) -> tuple[scipy.sparse.csr_matrix[float], list[list[int]]]: ...
    def __setstate__(
        self, arg: tuple[scipy.sparse.csr_matrix[float], Sequence[Sequence[int]]], /
    ) -> None: ...

def evaluate_list_vs_list(
    recommendations: Sequence[Sequence[int]],
    ground_truths: Sequence[Sequence[int]],
    n_items: int,
    n_threads: int,
) -> Metrics: ...
