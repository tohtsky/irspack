import os
import random
from typing import Optional, Tuple

import numpy as np

from ..definitions import InteractionMatrix
from ._util_cpp import (
    okapi_BM_25_weight,
    remove_diagonal,
    rowwise_train_test_split_d,
    rowwise_train_test_split_f,
    rowwise_train_test_split_i,
    sparse_mm_threaded,
    tf_idf_weight,
)


def rowwise_train_test_split(
    X: InteractionMatrix,
    test_ratio: float = 0.5,
    random_seed: Optional[int] = None,
) -> Tuple[InteractionMatrix, InteractionMatrix]:
    """Splits the non-zero elements of a sparse matrix into two (train & test interactions).
    For each row, the ratio of non-zero elements that become the test interaction
    is (approximately) constant.

    Args:
        X:
            The source sparse matrix.
        test_ratio:
            The ratio of test interactions for each row.
            That is, for each row, if it contains ``NNZ``-nonzero elements,
            the number of elements entering into the test interaction
            will be ``math.floor(test_ratio * NNZ)``.
            Defaults to 0.5.
        random_seed:
            The random seed. Defaults to None.

    Returns:
        A tuple of train & test interactions, which sum back to the original matrix.
    """
    if random_seed is None:
        random_seed = random.randint(-(2 ** 32), 2 ** 32 - 1)
    if X.dtype == np.float32:
        return rowwise_train_test_split_f(X, test_ratio, random_seed)
    elif X.dtype == np.float64:
        return rowwise_train_test_split_d(X, test_ratio, random_seed)
    elif X.dtype == np.int64:
        return rowwise_train_test_split_i(X, test_ratio, random_seed)
    else:
        original_dtype = X.dtype
        X_double = X.astype(np.float64)
        X_train_double, X_test_double = rowwise_train_test_split_d(
            X_double, test_ratio, random_seed
        )
        return (
            X_train_double.astype(original_dtype),
            X_test_double.astype(original_dtype),
        )


def get_n_threads(n_threads: Optional[int]) -> int:
    if n_threads is not None:
        return n_threads
    else:
        try:
            n_threads = int(os.environ.get("IRSPACK_NUM_THREADS_DEFAULT", "1"))
            return n_threads
        except:
            raise ValueError(
                'failed to interpret "IRSPACK_NUM_THREADS_DEFAULT" as an integer.'
            )


__all__ = [
    "rowwise_train_test_split",
    "sparse_mm_threaded",
    "okapi_BM_25_weight",
    "tf_idf_weight",
    "remove_diagonal",
]
