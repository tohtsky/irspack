m: int
n: int
from numpy import float32

import irspack._evaluator
from typing import *
from typing import Iterable as iterable
from typing import Iterator as iterator
from numpy import float64

_Shape = Tuple[int, ...]
import flags
import numpy
import scipy.sparse

__all__ = ["EvaluatorCore", "Metrics"]

class EvaluatorCore:
    def __init__(
        self,
        grount_truth: scipy.sparse.csr_matrix[float64],
        recommendable: List[List[int]],
    ) -> None: ...
    def get_ground_truth(self) -> scipy.sparse.csr_matrix[float64]: ...
    def get_metrics(
        self,
        score_array: numpy.ndarray[float64[m, n], flags.writeable, flags.c_contiguous],
        cutoff: int,
        offset: int,
        n_threads: int,
        recall_with_cutoff: bool = False,
    ) -> Metrics: ...
    pass

class Metrics:
    def __init__(self, arg0: int) -> None: ...
    def as_dict(self) -> Dict[str, float]: ...
    def merge(self, arg0: Metrics) -> None: ...
    pass
