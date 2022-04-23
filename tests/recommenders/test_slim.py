from typing import Dict

import numpy as np
import pytest
import scipy.sparse as sps

from irspack.recommenders import SLIMRecommender


def test_slim_positive(test_interaction_data: Dict[str, sps.csr_matrix]) -> None:
    try:
        from sklearn.linear_model import ElasticNet
    except ImportError:
        pytest.skip()
    alpha = 0.1
    l1_ratio = 0.5
    X = test_interaction_data["X_small"]
    rec = SLIMRecommender(
        X,
        alpha=alpha,
        l1_ratio=l1_ratio,
        positive_only=True,
        n_iter=100,
        n_threads=8,
        tol=0,
    )
    rec.learn()

    enet = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=False,
        positive=True,
        max_iter=100,
        tol=1e-8,
    )
    for iind in range(rec.W.shape[1]):
        m = rec.W[:, iind].toarray().ravel()
        Xcp = X.toarray()
        y = X[:, iind].toarray().ravel()
        Xcp[:, iind] = 0.0
        enet.fit(Xcp, y)
        np.testing.assert_allclose(enet.coef_, m, rtol=1e-2)


def test_slim_allow_negative(test_interaction_data: Dict[str, sps.csr_matrix]) -> None:
    try:
        from sklearn.linear_model import ElasticNet
    except ImportError:
        pytest.skip()

    ITER = 100
    alpha = 0.1
    l1_ratio = 0.5
    X = test_interaction_data["X_small"]
    rec = SLIMRecommender(
        X,
        alpha=alpha,
        l1_ratio=l1_ratio,
        positive_only=False,
        n_iter=ITER,
        n_threads=8,
        tol=0,
    )
    rec.learn()

    enet = ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=ITER, tol=1e-8
    )
    for iind in range(rec.W.shape[1]):
        m = rec.W[:, iind].toarray().ravel()
        Xcp = X.toarray()
        y = X[:, iind].toarray().ravel()
        Xcp[:, iind] = 0.0
        enet.fit(Xcp, y)
        np.testing.assert_allclose(enet.coef_, m, rtol=1e-2)


def test_slim_topk(test_interaction_data: Dict[str, sps.csr_matrix]) -> None:
    alpha = 0.0001
    l1_ratio = 0.0
    X = test_interaction_data["X_small"]
    rec = SLIMRecommender(
        X,
        alpha=alpha,
        l1_ratio=l1_ratio,
        positive_only=True,
        n_iter=100,
        n_threads=1,
        tol=0,
    )
    rec.learn()
    W_non_restricted = rec.W.toarray()

    rec_restricted = SLIMRecommender(
        X,
        alpha=alpha,
        l1_ratio=l1_ratio,
        positive_only=True,
        n_iter=100,
        n_threads=1,
        tol=0,
        top_k=1,
    )
    rec_restricted.learn()
    W_restricted = rec_restricted.W.toarray()
    for i in range(rec.n_items):
        gt = W_non_restricted[:, i]
        target = W_restricted[:, i]
        assert np.sum(target > 0) <= 1
        assert target.max() == pytest.approx(gt.max())
