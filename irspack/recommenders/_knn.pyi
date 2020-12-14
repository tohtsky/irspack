import irspack.recommenders._knn
from typing import *
from typing import Iterable as iterable
from typing import Iterator as iterator
from numpy import float64
_Shape = Tuple[int, ...]
import scipy.sparse
__all__  = [
"AsymmetricSimilarityComputer",
"CosineSimilarityComputer",
"JaccardSimilarityComputer"
]
class AsymmetricSimilarityComputer():
    def __init__(self, X: scipy.sparse.csr_matrix[float64], n_thread: int, shrinkage: float, alpha: float) -> None: ...
    def compute_block(self, arg0: scipy.sparse.csr_matrix[float64], arg1: int) -> scipy.sparse.csr_matrix[float64]: ...
    pass
class CosineSimilarityComputer():
    def __init__(self, X: scipy.sparse.csr_matrix[float64], n_thread: int, shrinkage: float) -> None: ...
    def compute_block(self, arg0: scipy.sparse.csr_matrix[float64], arg1: int) -> scipy.sparse.csr_matrix[float64]: ...
    pass
class JaccardSimilarityComputer():
    def __init__(self, X: scipy.sparse.csr_matrix[float64], n_thread: int, shrinkage: float) -> None: ...
    def compute_block(self, arg0: scipy.sparse.csr_matrix[float64], arg1: int) -> scipy.sparse.csr_matrix[float64]: ...
    pass
