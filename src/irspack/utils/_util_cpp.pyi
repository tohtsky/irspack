from collections.abc import Sequence
from typing import Annotated

import scipy.sparse
from numpy.typing import ArrayLike

def remove_diagonal(
    arg: scipy.sparse.csr_matrix[float], /
) -> scipy.sparse.csr_matrix[float]: ...
def sparse_mm_threaded(
    arg0: scipy.sparse.csr_matrix[float],
    arg1: scipy.sparse.csc_matrix[float],
    arg2: int,
    /,
) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="C")]: ...
def rowwise_train_test_split_by_ratio(
    arg0: scipy.sparse.csr_matrix[float], arg1: int, arg2: float, arg3: bool, /
) -> tuple[scipy.sparse.csr_matrix[float], scipy.sparse.csr_matrix[float]]: ...
def rowwise_train_test_split_by_fixed_n(
    arg0: scipy.sparse.csr_matrix[float], arg1: int, arg2: int, /
) -> tuple[scipy.sparse.csr_matrix[float], scipy.sparse.csr_matrix[float]]: ...
def okapi_BM_25_weight(
    X: scipy.sparse.csr_matrix[float], k1: float = 1.2, b: float = 0.75
) -> scipy.sparse.csr_matrix[float]: ...
def tf_idf_weight(
    X: scipy.sparse.csr_matrix[float], smooth: bool = True
) -> scipy.sparse.csr_matrix[float]: ...
def slim_weight_allow_negative(
    X: scipy.sparse.csr_matrix[float],
    n_threads: int,
    n_iter: int,
    l2_coeff: float,
    l1_coeff: float,
    tol: float,
    top_k: int = -1,
) -> scipy.sparse.csc_matrix[float]: ...
def slim_weight_positive_only(
    X: scipy.sparse.csr_matrix[float],
    n_threads: int,
    n_iter: int,
    l2_coeff: float,
    l1_coeff: float,
    tol: float,
    top_k: int = -1,
) -> scipy.sparse.csc_matrix[float]: ...
def retrieve_recommend_from_score_f64(
    score: Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="C")],
    allowed_indices: Sequence[Sequence[int]],
    cutoff: int,
    n_threads: int = 1,
) -> list[list[tuple[int, float]]]: ...
def retrieve_recommend_from_score_f32(
    score: Annotated[ArrayLike, dict(dtype="float32", shape=(None, None), order="C")],
    allowed_indices: Sequence[Sequence[int]],
    cutoff: int,
    n_threads: int = 1,
) -> list[list[tuple[int, float]]]: ...
