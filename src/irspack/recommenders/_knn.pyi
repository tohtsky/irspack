import scipy.sparse

class CosineSimilarityComputer:
    def __init__(
        self,
        X: scipy.sparse.csr_matrix[float],
        shrinkage: float,
        normalize: bool,
        n_threads: int = 1,
        max_chunk_size: int = 128,
    ) -> None: ...
    def compute_similarity(
        self, arg0: scipy.sparse.csr_matrix[float], arg1: int, /
    ) -> scipy.sparse.csr_matrix[float]: ...

class JaccardSimilarityComputer:
    def __init__(
        self,
        X: scipy.sparse.csr_matrix[float],
        shrinkage: float,
        n_threads: int = 1,
        max_chunk_size: int = 128,
    ) -> None: ...
    def compute_similarity(
        self, arg0: scipy.sparse.csr_matrix[float], arg1: int, /
    ) -> scipy.sparse.csr_matrix[float]: ...

class TverskyIndexComputer:
    def __init__(
        self,
        X: scipy.sparse.csr_matrix[float],
        shrinkage: float,
        alpha: float,
        beta: float,
        n_threads: int = 1,
        max_chunk_size: int = 128,
    ) -> None: ...
    def compute_similarity(
        self, arg0: scipy.sparse.csr_matrix[float], arg1: int, /
    ) -> scipy.sparse.csr_matrix[float]: ...

class AsymmetricSimilarityComputer:
    def __init__(
        self,
        X: scipy.sparse.csr_matrix[float],
        shrinkage: float,
        alpha: float,
        n_threads: int = 1,
        max_chunk_size: int = 128,
    ) -> None: ...
    def compute_similarity(
        self, arg0: scipy.sparse.csr_matrix[float], arg1: int, /
    ) -> scipy.sparse.csr_matrix[float]: ...

class P3alphaComputer:
    def __init__(
        self,
        X: scipy.sparse.csr_matrix[float],
        alpha: float = 0,
        n_threads: int = 1,
        max_chunk_size: int = 128,
    ) -> None: ...
    def compute_W(
        self, arg0: scipy.sparse.csr_matrix[float], arg1: int, /
    ) -> scipy.sparse.csc_matrix[float]: ...

class RP3betaComputer:
    def __init__(
        self,
        X: scipy.sparse.csr_matrix[float],
        alpha: float = 0,
        beta: float = 0,
        n_threads: int = 1,
        max_chunk_size: int = 128,
    ) -> None: ...
    def compute_W(
        self, arg0: scipy.sparse.csr_matrix[float], arg1: int, /
    ) -> scipy.sparse.csc_matrix[float]: ...
