m: int
n: int
from numpy import float32

import irspack.evaluator._core
import typing
import numpy
import scipy.sparse

_Shape = typing.Tuple[int, ...]

__all__ = ["EvaluatorCore", "Metrics"]

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
        score_array: numpy.ndarray[numpy.float32, _Shape[m, n]],
        cutoff: int,
        offset: int,
        n_threads: int,
        recall_with_cutoff: bool = False,
    ) -> Metrics: ...
    def get_metrics_f64(
        self,
        score_array: numpy.ndarray[numpy.float64, _Shape[m, n]],
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
