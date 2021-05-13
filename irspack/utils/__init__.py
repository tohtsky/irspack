from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sps

from irspack.definitions import InteractionMatrix, OptionalRandomState
from irspack.utils.id_mapping import IDMappedRecommender
from irspack.utils.random import convert_randomstate
from irspack.utils.threading import get_n_threads

from ._util_cpp import (
    okapi_BM_25_weight,
    remove_diagonal,
    rowwise_train_test_split_by_fixed_n,
    rowwise_train_test_split_by_ratio,
    sparse_mm_threaded,
    tf_idf_weight,
)


def l1_normalize_row(X: sps.csc_matrix) -> sps.csc_matrix:
    result: sps.csc_matrix = X.astype(np.float64)
    result.sort_indices()
    l1_norms: np.ndarray = result.sum(axis=1).A1
    result.data /= l1_norms[result.indices]
    return result


def rowwise_train_test_split(
    X: InteractionMatrix,
    test_ratio: float = 0.5,
    n_test: Optional[int] = None,
    ceil_n_test: bool = False,
    random_state: OptionalRandomState = None,
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
        random_state:
            The random state. Defaults to `None`.

    Returns:
        A tuple of train & test interactions, which sum back to the original matrix.
    """
    rns = convert_randomstate(random_state)
    random_seed = rns.randint(0, np.iinfo(np.int32).max, dtype=np.int32)
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


def df_to_sparse(
    df: pd.DataFrame,
    user_colname: str,
    item_colname: str,
    user_ids: Optional[List[Any]] = None,
    item_ids: Optional[List[Any]] = None,
    rating_colname: Optional[str] = None,
) -> Tuple[sps.csr_matrix, np.ndarray, np.ndarray]:
    r"""Convert pandas dataframe into sparse matrix.

    Args:
        df:
            The dataframe to be converted into a sparse matrix.
        user_colname:
            The column name for users.
        item_colname:
            The column name for items.
        user_ids:
            If not `None`, the resulting matrix's rows correspond exactly to this list.
        item_ids:
            If not `None`, the resulting matrix's columns correspond exactly to this list.
        rating_colname:
            If not `None`, the non-zero elements of the resulting matrix will correspond to the values of this column.
    Raises:
        RuntimeError:
            If `user_ids` is not `None` and `df[user_colname]` contains values not in `user_ids`.
        RuntimeError:
            If `item_ids` is not `None` and `df[item_colname]` contains values not in `item_ids`.

    Returns:
        - The resulting sparse matrix.
        - user ids corresponding to the rows in the matrix.
        - item ids corresponding to the columns in the matrix.
    """
    user_codes = pd.Categorical(df[user_colname], categories=user_ids)
    item_codes = pd.Categorical(df[item_colname], categories=item_ids)
    if np.any(user_codes.codes < 0):
        raise RuntimeError("Found unknown user id.")
    if np.any(item_codes.codes < 0):
        raise RuntimeError("Found unknown item id.")
    row = user_codes.codes
    unique_user_ids = user_codes.categories
    col = item_codes.codes
    unique_item_ids = item_codes.categories
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
    "l1_normalize_row",
    "rowwise_train_test_split",
    "sparse_mm_threaded",
    "okapi_BM_25_weight",
    "tf_idf_weight",
    "remove_diagonal",
    "get_n_threads",
    "IDMappedRecommender",
    "convert_randomstate",
]
