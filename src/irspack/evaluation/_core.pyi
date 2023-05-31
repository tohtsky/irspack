m: int
n: int
import typing

import numpy
import scipy.sparse
from numpy import float32

import irspack.evaluation._core

_Shape = typing.Tuple[int, ...]

__all__ = ["EvaluatorCore", "Metrics", "evaluate_list_vs_list"]

class EvaluatorCore:
    def __getstate__(self) -> tuple: ...
    def __init__(
        self,
        grount_truth: scipy.sparse.csr_matrix[numpy.float64],
        recommendable: typing.List[typing.List[int]],
    ) -> None: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def cache_X_as_set(self, arg0: int) -> None: ...
    def get_ground_truth(self) -> scipy.sparse.csr_matrix[numpy.float64]: ...
    def get_metrics_f32(
        self,
        score_array: numpy.ndarray[typing.Tuple[int, int], numpy.dtype[numpy.float32]],
        cutoff: int,
        offset: int,
        n_threads: int,
        recall_with_cutoff: bool = False,
    ) -> Metrics: ...
    def get_metrics_f64(
        self,
        score_array: numpy.ndarray[typing.Tuple[int, int], numpy.dtype[numpy.float64]],
        cutoff: int,
        offset: int,
        n_threads: int,
        recall_with_cutoff: bool = False,
    ) -> Metrics: ...
    pass

class Metrics:
    def __init__(self, arg0: int) -> None: ...
    def as_dict(self) -> typing.Dict[str, float]: ...
    def merge(self, arg0: Metrics) -> None: ...
    pass

def evaluate_list_vs_list(
    recomemndations: typing.List[typing.List[int]],
    grount_truths: typing.List[typing.List[int]],
    n_items: int,
    n_threads: int,
) -> Metrics:
    pass
