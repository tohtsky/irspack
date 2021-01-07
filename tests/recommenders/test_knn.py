import os

import numpy as np
import pytest
import scipy.sparse as sps

from irspack.recommenders.knn import (
    AsymmetricCosineKNNRecommender,
    CosineKNNRecommender,
    JaccardKNNRecommender,
    TverskyIndexKNNRecommender,
)
from irspack.recommenders.p3 import P3alphaRecommender
from irspack.recommenders.rp3 import RP3betaRecommender

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
    rec = CosineKNNRecommender(
        X, shrinkage=0, n_threads=5, top_k=X.shape[1], normalize=normalize
    )
    rec.learn()
    sim = rec.W.toarray()
    manual = X.T.toarray()  # I x U
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


@pytest.mark.parametrize("X", [X_many, X_small, X_many_dense])
def test_jaccard(X: sps.csr_matrix) -> None:
    rec = JaccardKNNRecommender(X, shrinkage=0, top_k=X.shape[1], n_threads=1)
    rec.learn()
    sim = rec.W.toarray()
    X_bin = X.copy()
    X_bin.sort_indices()
    X_bin.data[:] = 1
    manual = X_bin.T.toarray()  # I x U
    norm = manual.sum(axis=1)
    manual = manual.dot(manual.T)
    denom = norm[:, None] + norm[None, :] - manual + 1e-6
    denom[denom <= 1e-10] = 1e-10
    manual = manual / denom
    np.fill_diagonal(manual, 0)
    np.testing.assert_allclose(sim, manual)


@pytest.mark.parametrize(
    "X, alpha, shrinkage",
    [(X_many, 0.5, 0.0), (X_small, 0.7, 1.0), (X_many_dense, 0.01, 3)],
)
def test_asymmetric_cosine(X: sps.csr_matrix, alpha: float, shrinkage: float) -> None:
    rec = AsymmetricCosineKNNRecommender(
        X, shrinkage=shrinkage, alpha=alpha, n_threads=1, top_k=X.shape[1]
    )
    rec.learn()
    sim = rec.W.toarray()

    manual = X.T.toarray()  # I x U
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


@pytest.mark.parametrize(
    "X, alpha, beta, shrinkage",
    [(X_small, 0.7, 2, 0.0), (X_many, 0.5, 0.5, 0), (X_many_dense, 0.01, 0, 3)],
)
def test_tversky_index(
    X: sps.csr_matrix, alpha: float, beta: float, shrinkage: float
) -> None:
    RNS = np.random.RandomState(0)
    rec = TverskyIndexKNNRecommender(
        X, shrinkage=shrinkage, alpha=alpha, beta=beta, n_threads=1, top_k=X.shape[1]
    )
    rec.learn()
    sim = rec.W.toarray()
    tested_index_row = RNS.randint(0, sim.shape[0], size=100)
    tested_index_col = RNS.randint(0, sim.shape[0], size=100)
    X_csc = X.tocsc()
    X_csc.sorted_indices()
    for i, j in zip(tested_index_row, tested_index_col):
        if i == j:
            continue
        computed = sim[i, j]
        U_i = set(X_csc[:, i].nonzero()[0])
        U_j = set(X_csc[:, j].nonzero()[0])
        intersect = U_i.intersection(U_j)
        Ui_minus_Uj = U_i.difference(U_j)
        Uj_minus_Ui = U_j.difference(U_i)
        target = len(intersect) / (
            len(intersect)
            + alpha * len(Ui_minus_Uj)
            + beta * len(Uj_minus_Ui)
            + shrinkage
            + 1e-6
        )
        assert computed == pytest.approx(target)


@pytest.mark.parametrize("X", [X_many, X_small])
def test_topk(X: sps.csr_matrix) -> None:
    rec = CosineKNNRecommender(X, shrinkage=0, top_k=30, n_threads=5)
    rec.learn()
    sim = rec.W.toarray()
    assert np.all((sim > 0).sum(axis=1) <= 30)


@pytest.mark.parametrize("X, alpha", [(X_small, 0.001), (X_many_dense, 2), (X_many, 1)])
def test_p3(X: sps.csr_matrix, alpha: float) -> None:
    rec = P3alphaRecommender(X, alpha=alpha, n_threads=4)
    rec.learn()
    W = rec.W.toarray()

    P_ui = np.power(X.toarray(), alpha)
    P_iu = np.power(X.T.toarray(), alpha)

    def zero_or_1(X: np.ndarray) -> np.ndarray:
        X = X.copy()
        X[X == 0] = 1
        return X

    P_ui /= zero_or_1(P_ui.sum(axis=1))[:, None]
    P_iu /= zero_or_1(P_iu.sum(axis=1))[:, None]
    W_man = P_iu.dot(P_ui)
    np.testing.assert_allclose(W, W_man)

    rec_norm = P3alphaRecommender(
        X, alpha=alpha, n_threads=4, top_k=2, normalize_weight=True
    )
    rec_norm.learn()
    # rec
    W_sum = rec_norm.W.sum(axis=1).A1
    for w in W_sum:
        assert w == pytest.approx(1.0)


@pytest.mark.parametrize(
    "X, alpha, beta", [(X_small, 0.001, 3), (X_many_dense, 2, 5), (X_many, 1, 0.2)]
)
def test_rp3(X: sps.csr_matrix, alpha: float, beta: float) -> None:
    rec = RP3betaRecommender(X, alpha=alpha, beta=beta, n_threads=4)
    rec.learn()
    W = rec.W.toarray()
    W_sum = W.sum(axis=1)
    W_sum = W_sum[W_sum >= 0]

    popularity = X.sum(axis=0).A1.ravel() ** beta

    def zero_or_1(X: np.ndarray) -> np.ndarray:
        X = X.copy()
        X[X == 0] = 1
        return X

    P_ui = np.power(X.toarray(), alpha)
    P_iu = np.power(X.T.toarray(), alpha)

    P_ui /= zero_or_1(P_ui.sum(axis=1))[:, None]
    P_iu /= zero_or_1(P_iu.sum(axis=1))[:, None]
    W_man = P_iu.dot(P_ui)

    # p_{ui} ^{RP3} = p_{ui} ^{P3} / popularity_i ^ beta
    from sklearn.preprocessing import normalize

    W_man = W_man / zero_or_1(popularity)[None, :]
    # W_man = normalize(W_man, axis=1, norm="l1")
    # print(W_man.sum(axis=1))
    np.testing.assert_allclose(W, W_man)

    rec_norm = RP3betaRecommender(
        X, alpha=alpha, n_threads=4, top_k=2, normalize_weight=True
    )
    rec_norm.learn()
    # rec
    W_sum = rec_norm.W.sum(axis=1).A1
    for w in W_sum:
        assert w == pytest.approx(1.0)


def test_raise_shrinkage() -> None:
    with pytest.raises(ValueError):
        _ = P3alphaRecommender(X_many.T, alpha=1.0, n_threads=0)
        _.learn()

    with pytest.raises(ValueError):
        _ = P3alphaRecommender(X_many.T, alpha=-1.0, n_threads=1)
        _.learn()

    with pytest.raises(ValueError):
        rec = AsymmetricCosineKNNRecommender(X_many, 0, alpha=1.5)
        rec.learn()


@pytest.mark.parametrize(
    "X",
    [X_small],
)
def test_n_threads(X: sps.csr_matrix) -> None:
    os.environ["IRSPACK_NUM_THREADS_DEFAULT"] = "31"
    rec = CosineKNNRecommender(X)
    assert rec.n_threads == 31
    os.environ["IRSPACK_NUM_THREADS_DEFAULT"] = "NOT_A_INTEGER"
    with pytest.raises(ValueError):
        rec = CosineKNNRecommender(X)
    os.environ.pop("IRSPACK_NUM_THREADS_DEFAULT")
    rec = CosineKNNRecommender(X)
    assert rec.n_threads == 1
