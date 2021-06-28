m: int
n: int
from numpy import float32

import irspack.utils._util_cpp
import typing
import numpy
import scipy.sparse

_Shape = typing.Tuple[int, ...]

__all__ = [
    "okapi_BM_25_weight",
    "remove_diagonal",
    "retrieve_recommend_from_score_f32",
    "retrieve_recommend_from_score_f64",
    "rowwise_train_test_split_by_fixed_n",
    "rowwise_train_test_split_by_ratio",
    "slim_weight_allow_negative",
    "slim_weight_positive_only",
    "sparse_mm_threaded",
    "tf_idf_weight",
]

def okapi_BM_25_weight(
    X: scipy.sparse.csr_matrix[numpy.float64], k1: float = 1.2, b: float = 0.75
) -> scipy.sparse.csr_matrix[numpy.float64]:
    pass

def remove_diagonal(
    arg0: scipy.sparse.csr_matrix[numpy.float64],
) -> scipy.sparse.csr_matrix[numpy.float64]:
    pass

def retrieve_recommend_from_score_f32(
    score: numpy.ndarray[numpy.float32, _Shape[m, n]],
    allowed_indices: typing.List[typing.List[int]],
    cutoff: int,
    n_threads: int = 1,
) -> typing.List[typing.List[typing.Tuple[int, float]]]:
    pass

def retrieve_recommend_from_score_f64(
    score: numpy.ndarray[numpy.float64, _Shape[m, n]],
    allowed_indices: typing.List[typing.List[int]],
    cutoff: int,
    n_threads: int = 1,
) -> typing.List[typing.List[typing.Tuple[int, float]]]:
    pass

def rowwise_train_test_split_by_fixed_n(
    arg0: scipy.sparse.csr_matrix[numpy.float64], arg1: int, arg2: int
) -> typing.Tuple[
    scipy.sparse.csr_matrix[numpy.float64], scipy.sparse.csr_matrix[numpy.float64]
]:
    pass

def rowwise_train_test_split_by_ratio(
    arg0: scipy.sparse.csr_matrix[numpy.float64], arg1: int, arg2: float, arg3: bool
) -> typing.Tuple[
    scipy.sparse.csr_matrix[numpy.float64], scipy.sparse.csr_matrix[numpy.float64]
]:
    pass

def slim_weight_allow_negative(
    X: scipy.sparse.csr_matrix[numpy.float32],
    n_threads: int,
    n_iter: int,
    l2_coeff: float,
    l1_coeff: float,
    tol: float,
    top_k: int = -1,
) -> scipy.sparse.csc_matrix[numpy.float32]:
    pass

def slim_weight_positive_only(
    X: scipy.sparse.csr_matrix[numpy.float32],
    n_threads: int,
    n_iter: int,
    l2_coeff: float,
    l1_coeff: float,
    tol: float,
    top_k: int = -1,
) -> scipy.sparse.csc_matrix[numpy.float32]:
    pass

def sparse_mm_threaded(
    arg0: scipy.sparse.csr_matrix[numpy.float64],
    arg1: scipy.sparse.csc_matrix[numpy.float64],
    arg2: int,
) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
    pass

def tf_idf_weight(
    X: scipy.sparse.csr_matrix[numpy.float64], smooth: bool = True
) -> scipy.sparse.csr_matrix[numpy.float64]:
    pass
