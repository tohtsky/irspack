m: int
n: int
from numpy import float32

import irspack.recommenders._knn
import typing
import numpy
import scipy.sparse

_Shape = typing.Tuple[int, ...]

__all__ = [
    "AsymmetricSimilarityComputer",
    "CosineSimilarityComputer",
    "JaccardSimilarityComputer",
    "P3alphaComputer",
    "RP3betaComputer",
    "TverskyIndexComputer",
]

class AsymmetricSimilarityComputer:
    def __init__(
        self,
        X: scipy.sparse.csr_matrix[numpy.float64],
        shrinkage: float,
        alpha: float,
        n_threads: int = 1,
        max_chunk_size: int = 128,
    ) -> None: ...
    def compute_similarity(
        self, arg0: scipy.sparse.csr_matrix[numpy.float64], arg1: int
    ) -> scipy.sparse.csr_matrix[numpy.float64]: ...
    pass

class CosineSimilarityComputer:
    def __init__(
        self,
        X: scipy.sparse.csr_matrix[numpy.float64],
        shrinkage: float,
        normalize: bool,
        n_threads: int = 1,
        max_chunk_size: int = 128,
    ) -> None: ...
    def compute_similarity(
        self, arg0: scipy.sparse.csr_matrix[numpy.float64], arg1: int
    ) -> scipy.sparse.csr_matrix[numpy.float64]: ...
    pass

class JaccardSimilarityComputer:
    def __init__(
        self,
        X: scipy.sparse.csr_matrix[numpy.float64],
        shrinkage: float,
        n_threads: int = 1,
        max_chunk_size: int = 128,
    ) -> None: ...
    def compute_similarity(
        self, arg0: scipy.sparse.csr_matrix[numpy.float64], arg1: int
    ) -> scipy.sparse.csr_matrix[numpy.float64]: ...
    pass

class P3alphaComputer:
    def __init__(
        self,
        X: scipy.sparse.csr_matrix[numpy.float64],
        alpha: float = 0,
        n_threads: int = 1,
        max_chunk_size: int = 128,
    ) -> None: ...
    def compute_W(
        self, arg0: scipy.sparse.csr_matrix[numpy.float64], arg1: int
    ) -> scipy.sparse.csc_matrix[numpy.float64]: ...
    pass

class RP3betaComputer:
    def __init__(
        self,
        X: scipy.sparse.csr_matrix[numpy.float64],
        alpha: float = 0,
        beta: float = 0,
        n_threads: int = 1,
        max_chunk_size: int = 128,
    ) -> None: ...
    def compute_W(
        self, arg0: scipy.sparse.csr_matrix[numpy.float64], arg1: int
    ) -> scipy.sparse.csc_matrix[numpy.float64]: ...
    pass

class TverskyIndexComputer:
    def __init__(
        self,
        X: scipy.sparse.csr_matrix[numpy.float64],
        shrinkage: float,
        alpha: float,
        beta: float,
        n_threads: int = 1,
        max_chunk_size: int = 128,
    ) -> None: ...
    def compute_similarity(
        self, arg0: scipy.sparse.csr_matrix[numpy.float64], arg1: int
    ) -> scipy.sparse.csr_matrix[numpy.float64]: ...
    pass
