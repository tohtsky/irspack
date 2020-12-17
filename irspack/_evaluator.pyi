import irspack._evaluator
from typing import *
from typing import Iterable as iterable
from typing import Iterator as iterator
from numpy import float64
_Shape = Tuple[int, ...]
import flags
import numpy
import scipy.sparse
__all__  = [
"EvaluatorCore",
"Metrics"
]
class EvaluatorCore():
    def __init__(self, arg0: scipy.sparse.csr_matrix[float64]) -> None: ...
    def get_metrics(self, arg0: numpy.ndarray[float64[m, n], flags.writeable, flags.c_contiguous], arg1: int, arg2: int, arg3: int, arg4: bool) -> Metrics: ...
    pass
class Metrics():
    def __init__(self, arg0: int) -> None: ...
    def as_dict(self) -> Dict[str, float]: ...
    def merge(self, arg0: Metrics) -> None: ...
    pass
