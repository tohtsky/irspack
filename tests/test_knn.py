import numpy as np
import pytest
import scipy.sparse as sps
from irspack.recommenders._knn import (
    CosineSimilarityComputer,
    JaccardSimilarityComputer,
    AsymmetricSimilarityComputer,
)

X_small = sps.csr_matrix(
    np.asfarray([[1, 1, 2, 3, 4], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
)
X_many = np.random.rand(888, 512)
X_many[X_many <= 0.9] = 0
X_many[X_many > 0.9] = 1
X_many = sps.csr_matrix(X_many)
X_many.sort_indices()


@pytest.mark.parametrize("X", [X_many, X_small])
def test_cosine(X):
    computer = CosineSimilarityComputer(X.T, 0, 5)
    sim = computer.compute_similarity(X.T, X.shape[1]).toarray()
    manual = X.T.toarray()  # I x U
    norm = (manual ** 2).sum(axis=1) ** 0.5
    manual = manual / norm[:, None]
    manual = manual.dot(manual.T)
    np.testing.assert_allclose(
        sim,
        manual,
    )


@pytest.mark.parametrize("X", [X_many, X_small])
def test_jaccard(X):
    computer = JaccardSimilarityComputer(X.T, 0, 5)
    X_bin = X.copy()
    X_bin.sort_indices()
    X_bin.data[:] = 1
    sim = computer.compute_similarity(X.T, X.shape[1]).toarray()
    manual = X_bin.T.toarray()  # I x U
    norm = manual.sum(axis=1)
    manual = manual.dot(manual.T)
    denom = norm[:, None] + norm[None, :] - manual
    denom[denom <= 1e-10] = 1e-10
    manual = manual / denom
    assert np.all(np.abs(sim - manual) <= 1e-5)


@pytest.mark.parametrize("X, alpha", [(X_many, 0.3), (X_small, 0.7)])
def test_asymmetric_cosine(X, alpha):
    computer = AsymmetricSimilarityComputer(X.T, 0, alpha, 1)
    sim = computer.compute_similarity(X.T, X.shape[1]).toarray()
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
    computer = CosineSimilarityComputer(X.T, 0, 5)
    sim = computer.compute_similarity(X.T, 30).toarray()
    assert np.all((sim > 0).sum(axis=1) <= 30)
