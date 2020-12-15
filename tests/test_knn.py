import numpy as np
import pytest
import scipy.sparse as sps
from irspack.recommenders.knn import (
    CosineKNNRecommender,
    JaccardKNNRecommender,
    AssumetricCosineKNNRecommender,
)

X_small = sps.csr_matrix(
    np.asfarray([[1, 1, 2, 3, 4], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
)
X_many = np.random.rand(888, 512)
X_many[X_many <= 0.9] = 0
X_many[X_many > 0.9] = 1
X_many = sps.csr_matrix(X_many)
X_many.sort_indices()

X_many_dense = sps.csr_matrix(np.random.rand(133, 245))


@pytest.mark.parametrize("X", [X_many, X_small, X_many_dense])
def test_cosine(X):
    rec = CosineKNNRecommender(X, shrinkage=0, n_thread=5, top_k=X.shape[1])
    rec.learn()
    sim = rec.W.toarray()
    manual = X.T.toarray()  # I x U
    norm = (manual ** 2).sum(axis=1) ** 0.5
    manual = manual / norm[:, None]
    manual = manual.dot(manual.T)
    np.testing.assert_allclose(
        sim,
        manual,
    )


@pytest.mark.parametrize("X", [X_many, X_small, X_many_dense])
def test_jaccard(X):
    rec = JaccardKNNRecommender(X, shrinkage=0, top_k=X.shape[1], n_thread=1)
    rec.learn()
    sim = rec.W.toarray()
    X_bin = X.copy()
    X_bin.sort_indices()
    X_bin.data[:] = 1
    manual = X_bin.T.toarray()  # I x U
    norm = manual.sum(axis=1)
    manual = manual.dot(manual.T)
    denom = norm[:, None] + norm[None, :] - manual
    denom[denom <= 1e-10] = 1e-10
    manual = manual / denom
    assert np.all(np.abs(sim - manual) <= 1e-5)


@pytest.mark.parametrize(
    "X, alpha", [(X_many, 0.5), (X_small, 0.7), (X_many_dense, (0.01))]
)
def test_asymmetric_cosine(X, alpha):
    rec = AssumetricCosineKNNRecommender(
        X, shrinkage=0, alpha=alpha, n_thread=1, top_k=X.shape[1]
    )
    rec.learn()
    sim = rec.W.toarray()

    manual = X.T.toarray()  # I x U
    norm = (manual ** 2).sum(axis=1)
    norm_alpha = np.power(norm, alpha)
    norm_1malpha = np.power(norm, 1 - alpha)
    manual_sim = manual.dot(manual.T)
    manual_sim /= norm_alpha[:, None]
    manual_sim /= norm_1malpha[None, :]
    np.testing.assert_allclose(
        sim,
        manual_sim,
    )


@pytest.mark.parametrize("X", [X_many, X_small])
def test_topk(X):
    rec = CosineKNNRecommender(X, shrinkage=0, top_k=30, n_thread=5)
    rec.learn()
    sim = rec.W.toarray()
    assert np.all((sim > 0).sum(axis=1) <= 30)
