import numpy as np
import scipy.sparse as sps
from irspack.recommenders._knn import (
    CosineSimilarityComputer,
    JaccardSimilarityComputer,
)

X = sps.csr_matrix(
    np.asfarray(
        [[1, 1, 2, 3, 4], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]
    )
)
X_bin = X.copy()
X_bin.sort_indices()
X_bin.data[:] = 1


def test_cosine():
    computer = CosineSimilarityComputer(X.T, 0, 5)
    sim = computer.compute_similarity(X.T, 100).toarray()
    manual = X.T.toarray()  # I x U
    norm = (manual ** 2).sum(axis=1) ** 0.5
    manual = manual / norm[:, None]
    manual = manual.dot(manual.T)
    assert np.all(np.abs(sim - manual) <= 1e-5)


def test_cosine_topK():
    computer = CosineSimilarityComputer(X.T, 0, 5)
    sim = computer.compute_similarity(X.T, 1).toarray()
    assert np.all(np.bincount(sim.nonzero()[0]) <= 1)


def test_jaccard():
    computer = JaccardSimilarityComputer(X.T, 0, 5)
    sim = computer.compute_similarity(X.T, 100).toarray()
    manual = X_bin.T.toarray()  # I x U
    norm = manual.sum(axis=1)
    manual = manual.dot(manual.T)
    denom = norm[:, None] + norm[None, :] - manual
    denom[denom <= 1e-10] = 1e-10
    print(sim)
    manual = manual / denom
    print()
    print(manual)
    # assert np.all(np.abs(sim - manual) <= 1e-5)
