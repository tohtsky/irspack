"""Backend C++ inplementation for Random walk with restart."""
from typing import Iterable as iterable
from typing import Iterator as iterator
from typing import *

from numpy import float32, float64, int32

_Shape = Tuple[int, ...]
import scipy.sparse

__all__ = ["RandomWalkGenerator"]

class RandomWalkGenerator:
    def __init__(self, arg0: scipy.sparse.csr_matrix[float32]) -> None: ...
    def run_with_restart(
        self, arg0: float, arg1: int, arg2: int, arg3: int, arg4: int
    ) -> scipy.sparse.csr_matrix[int32]: ...
    pass
