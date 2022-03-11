import numpy as np
import pytest
import scipy.sparse as sps

from irspack.utils import (
    okapi_BM_25_weight,
    rowwise_train_test_split,
    sparse_mm_threaded,
    tf_idf_weight,
)

X = sps.csr_matrix(
    np.asfarray([[1, 1, 2, 3, 4], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
)
X.sort_indices()
X_array = X.toarray()


def test_sparse_mm_threaded() -> None:
    X_mm = sparse_mm_threaded(X, X.T, 3)
    X_sps = X.dot(X.T).toarray()
    np.testing.assert_allclose(X_mm, X_sps)


def test_split() -> None:
    X_1, X_2 = rowwise_train_test_split(X, test_ratio=0.5, random_state=1)
    np.testing.assert_allclose(X.toarray(), (X_1 + X_2).toarray())

    # should have no overwrap
    assert np.all(X_1.multiply(X_2).toarray() == 0)
    nnzs = X.indptr[1:] - X.indptr[:-1]
    X_2_bin = X_2.copy()
    X_2_bin.data[:] = 1.0
    X_2_nnzs = X_2_bin.sum(axis=1).A1
    assert np.all((nnzs * 0.5) >= X_2_nnzs)


def test_split_ceil() -> None:
    X_1, X_2 = rowwise_train_test_split(
        X, test_ratio=0.5, random_state=1, ceil_n_heldout=True
    )
    np.testing.assert_allclose(X.toarray(), (X_1 + X_2).toarray())

    # should have no overwrap
    assert np.all(X_1.multiply(X_2).toarray() == 0)
    nnzs = X.indptr[1:] - X.indptr[:-1]
    X_2_bin = X_2.copy()
    X_2_bin.data[:] = 1.0
    X_2_nnzs = X_2_bin.sum(axis=1).A1
    assert np.all((nnzs * 0.5) <= X_2_nnzs)


def test_split_fixed_n() -> None:
    X_1, X_2 = rowwise_train_test_split(X, test_ratio=0.5, n_test=1, random_state=1)
    np.testing.assert_allclose(X.toarray(), (X_1 + X_2).toarray())

    # should have no overwrap
    assert np.all(X_1.multiply(X_2).toarray() == 0)
    X_2.data[:] = 1
    assert X_2.sum(axis=1).max() <= 1


def test_bm25() -> None:
    k1 = 1.4
    b = 0.8
    X_weighted = okapi_BM_25_weight(X, k1=k1, b=b).toarray()

    df = (X.toarray() > 0).sum(axis=0).ravel()
    idf = np.log(X.shape[0] / (df + 1) + 1)
    avgdl = X_array.sum(axis=1).mean()
    for row in range(X.shape[0]):
        for col in range(X.shape[1]):
            tf = X_array[row, col]
            py_answer = (
                idf[col]
                * (tf * (k1 + 1))
                / (tf + k1 * (1 - b + b * X_array[row].sum() / avgdl))
            )
            assert py_answer == pytest.approx(X_weighted[row, col])


def test_tf_idf() -> None:
    X_manual = (
        X.toarray() * np.log(X.shape[0] / (1 + np.bincount(X.nonzero()[1])))[None, :]
    )
    X_tf_idf = tf_idf_weight(X).toarray()
    np.testing.assert_allclose(X_manual, X_tf_idf)
