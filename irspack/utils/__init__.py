import os
import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sps

from ..definitions import InteractionMatrix
from ._util_cpp import (
    okapi_BM_25_weight,
    remove_diagonal,
    rowwise_train_test_split_by_fixed_n,
    rowwise_train_test_split_by_ratio,
    sparse_mm_threaded,
    tf_idf_weight,
)


def rowwise_train_test_split(
    X: InteractionMatrix,
    test_ratio: float = 0.5,
    n_test: Optional[int] = None,
    ceil_n_test: bool = False,
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
    original_dtype = X.dtype
    X_double = X.astype(np.float64)
    if n_test is None:
        X_train_double, X_test_double = rowwise_train_test_split_by_ratio(
            X_double, random_seed, test_ratio, ceil_n_test
        )
    else:
        X_train_double, X_test_double = rowwise_train_test_split_by_fixed_n(
            X_double, random_seed, n_test
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
            n_threads_cand = os.environ.get(
                "IRSPACK_NUM_THREADS_DEFAULT", os.cpu_count()
            )
            return int(n_threads_cand or 1)
        except:
            raise ValueError(
                'failed to interpret "IRSPACK_NUM_THREADS_DEFAULT" as an integer.'
            )


def df_to_sparse(
    df: pd.DataFrame,
    user_colname: str,
    item_colname: str,
    rating_colname: Optional[str] = None,
) -> Tuple[sps.csr_matrix, np.ndarray, np.ndarray]:
    row, unique_user_ids = pd.factorize(df[user_colname], sort=True)
    col, unique_item_ids = pd.factorize(df[item_colname], sort=True)
    if rating_colname is None:
        data = np.ones(df.shape[0])
    else:
        data = np.asfarray(df[rating_colname].values)
    return (
        sps.csr_matrix(
            (data, (row, col)), shape=(len(unique_user_ids), len(unique_item_ids))
        ),
        unique_user_ids,
        unique_item_ids,
    )


__all__ = [
    "rowwise_train_test_split",
    "sparse_mm_threaded",
    "okapi_BM_25_weight",
    "tf_idf_weight",
    "remove_diagonal",
]
