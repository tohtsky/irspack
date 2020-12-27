import numpy as np
import pytest
import scipy.sparse as sps

from irspack.recommenders import IALSRecommender

X_small = sps.csr_matrix(
    np.asfarray([[6, 1, 2, 3, 4], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]])
)


def test_ials_overfit_cholesky() -> None:
    X = X_small
    rec = IALSRecommender(X, n_components=4, alpha=0, reg=0, use_cg=False)
    rec.learn()
    assert rec.trainer is not None
    uvec = rec.trainer.core_trainer.transform_user(X.tocsr().astype(np.float32))
    ivec = rec.trainer.core_trainer.transform_item(X.tocsr().astype(np.float32))
    X = X.toarray()
    reprod = uvec.dot(ivec.T)
    np.testing.assert_allclose(reprod, X, rtol=1e-2, atol=1e-2)


def test_ials_overfit_cg() -> None:
    X = X_small
    rec = IALSRecommender(
        X, n_components=4, alpha=0, reg=1e-2, use_cg=True, max_cg_steps=4
    )
    rec.learn()
    assert rec.trainer is not None
    uvec = rec.trainer.core_trainer.transform_user(X.tocsr().astype(np.float32))
    ivec = rec.trainer.core_trainer.transform_item(X.tocsr().astype(np.float32))
    X = X.toarray()
    reprod = uvec.dot(ivec.T)
    np.testing.assert_allclose(reprod, X, rtol=1e-2, atol=1e-2)


@pytest.mark.xfail
def test_ials_cg_underfit() -> None:
    X = X_small
    rec = IALSRecommender(
        X,
        n_components=4,
        alpha=0,
        reg=1e-2,
        use_cg=True,
        max_cg_steps=1,
        max_epoch=10,
    )
    rec.learn()
    assert rec.trainer is not None
    uvec = rec.trainer.core_trainer.user
    ivec = rec.trainer.core_trainer.item
    X = X.toarray()
    reprod = uvec.dot(ivec.T)
    np.testing.assert_allclose(reprod, X, rtol=1e-2, atol=1e-2)
