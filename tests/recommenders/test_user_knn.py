import numpy as np
import pytest
import scipy.sparse as sps

from irspack.recommenders.user_knn import (
    AsymmetricCosineUserKNNRecommender,
    CosineUserKNNRecommender,
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


@pytest.mark.parametrize(
    "X, normalize", [(X_many, True), (X_small, False), (X_many_dense, True)]
)
def test_cosine(X: sps.csr_matrix, normalize: bool) -> None:
    rec = CosineUserKNNRecommender(
        X, shrinkage=0, n_threads=5, top_k=X.shape[0], normalize=normalize
    )
    with pytest.raises(RuntimeError):
        U = rec.U
    rec.learn()
    sim = rec.U.toarray()
    manual = X.toarray()  # U x I
    norm = (manual ** 2).sum(axis=1) ** 0.5
    manual = manual.dot(manual.T)
    if normalize:
        denom = norm[:, None] * norm[None, :] + 1e-6
        manual /= denom
    np.fill_diagonal(manual, 0)
    np.testing.assert_allclose(
        sim,
        manual,
    )


@pytest.mark.parametrize(
    "X, alpha, shrinkage",
    [(X_many, 0.5, 0.0), (X_small, 0.7, 1.0), (X_many_dense, 0.01, 3)],
)
def test_asymmetric_cosine(X: sps.csr_matrix, alpha: float, shrinkage: float) -> None:
    rec = AsymmetricCosineUserKNNRecommender(
        X, shrinkage=shrinkage, alpha=alpha, n_threads=1, top_k=X.shape[0]
    )
    rec.learn()
    sim = rec.U.toarray()

    manual = X.toarray()
    norm = (manual ** 2).sum(axis=1)
    norm_alpha = np.power(norm, alpha)
    norm_1malpha = np.power(norm, 1 - alpha)
    manual_sim = manual.dot(manual.T)
    denom = norm_alpha[:, None] * norm_1malpha[None, :] + 1e-6 + shrinkage
    manual_sim /= denom
    np.fill_diagonal(manual_sim, 0)
    np.testing.assert_allclose(
        sim,
        manual_sim,
    )


@pytest.mark.parametrize("X", [X_many, X_small])
def test_topk(X: sps.csr_matrix) -> None:
    rec = AsymmetricCosineUserKNNRecommender(X, shrinkage=0, top_k=30, n_threads=5)
    rec.learn()
    sim = rec.U.toarray()
    assert np.all((sim > 0).sum(axis=1) <= 30)
