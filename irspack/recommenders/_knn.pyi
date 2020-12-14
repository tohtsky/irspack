from typing import *
from typing import Iterable as iterable
from typing import Iterator as iterator
from numpy import float64

_Shape = Tuple[int, ...]
import scipy.sparse

__all__ = ["CosineKNNComputer"]


class CosineKNNComputer:
    def __init__(
        self, arg0: scipy.sparse.csr_matrix[float64], arg1: int, arg2: float
    ) -> None:
        ...

    def compute_block(
        self, arg0: scipy.sparse.csr_matrix[float64], arg1: int
    ) -> scipy.sparse.csr_matrix[float64]:
        ...

    pass
