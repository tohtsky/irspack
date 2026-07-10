from collections.abc import Sequence
from typing import Annotated

import numpy
import scipy
from numpy.typing import NDArray

def remove_diagonal(
    X: scipy.sparse.csr_matrix[float],
) -> scipy.sparse.csr_matrix[float]: ...
def sparse_mm_threaded(
    left: scipy.sparse.csr_matrix[float],
    right: scipy.sparse.csc_matrix[float],
    n_threads: int,
) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C")]: ...
def rowwise_train_test_split_by_ratio(
    X: scipy.sparse.csr_matrix[float],
    random_seed: int,
    heldout_ratio: float,
    n_test_ceil: bool,
) -> tuple[scipy.sparse.csr_matrix[float], scipy.sparse.csr_matrix[float]]: ...
def rowwise_train_test_split_by_fixed_n(
    X: scipy.sparse.csr_matrix[float], random_seed: int, n_held_out: int
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
    score: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C")],
    allowed_indices: Sequence[Sequence[int]],
    cutoff: int,
    n_threads: int = 1,
) -> list[list[tuple[int, float]]]: ...
def retrieve_recommend_from_score_f32(
    score: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")],
    allowed_indices: Sequence[Sequence[int]],
    cutoff: int,
    n_threads: int = 1,
) -> list[list[tuple[int, float]]]: ...
