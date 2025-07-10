from collections.abc import Sequence
from typing import Annotated

import scipy.sparse
from numpy.typing import ArrayLike

class Metrics:
    def __init__(self, arg: int, /) -> None: ...
    def merge(self, arg: Metrics, /) -> None: ...
    def as_dict(self) -> dict[str, float]: ...

class EvaluatorCore:
    def __init__(
        self,
        grount_truth: scipy.sparse.csr_matrix[float],
        recommendable: Sequence[Sequence[int]],
    ) -> None: ...
    def get_metrics_f64(
        self,
        score_array: Annotated[ArrayLike, dict(dtype="float64", shape=(None, None))],
        cutoff: int,
        offset: int,
        n_threads: int,
        recall_with_cutoff: bool = False,
    ) -> Metrics: ...
    def get_metrics_f32(
        self,
        score_array: Annotated[ArrayLike, dict(dtype="float32", shape=(None, None))],
        cutoff: int,
        offset: int,
        n_threads: int,
        recall_with_cutoff: bool = False,
    ) -> Metrics: ...
    def get_ground_truth(self) -> scipy.sparse.csr_matrix[float]: ...
    def cache_X_as_set(self, arg: int, /) -> None: ...
    def __getstate__(
        self,
    ) -> tuple[scipy.sparse.csr_matrix[float], list[list[int]]]: ...
    def __setstate__(
        self, arg: tuple[scipy.sparse.csr_matrix[float], Sequence[Sequence[int]]], /
    ) -> None: ...

def evaluate_list_vs_list(
    recomemndations: Sequence[Sequence[int]],
    grount_truths: Sequence[Sequence[int]],
    n_items: int,
    n_threads: int,
) -> Metrics: ...
