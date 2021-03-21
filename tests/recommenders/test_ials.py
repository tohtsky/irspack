from typing import Dict, Tuple

import numpy as np
import pytest
import scipy.sparse as sps
from scipy.optimize import minimize

from irspack.recommenders import IALSRecommender


def test_ials_overfit_cholesky(
    test_interaction_data: Dict[str, sps.csr_matrix]
) -> None:
    X = test_interaction_data["X_small"]
    rec = IALSRecommender(
        X,
        n_components=4,
        alpha=0,
        reg=0.001,
        use_cg=False,
    )
    rec.learn()
    assert rec.trainer is not None
    uvec = rec.compute_user_embedding(X)
    ivec = rec.compute_item_embedding(X)
    X = X.toarray()
    X[X.nonzero()] = 1.0
    reprod = uvec.dot(ivec.T)
    np.testing.assert_allclose(reprod, X, rtol=1e-2, atol=1e-2)


def test_ials_overfit_cg(test_interaction_data: Dict[str, sps.csr_matrix]) -> None:
    X = test_interaction_data["X_small"]
    rec = IALSRecommender(
        X, n_components=4, alpha=0, reg=1e-2, use_cg=True, max_cg_steps=4
    )
    rec.learn()
    assert rec.trainer is not None
    uvec = rec.trainer.core_trainer.transform_user(X.tocsr().astype(np.float32))
    ivec = rec.trainer.core_trainer.transform_item(X.tocsr().astype(np.float32))
    X_dense = X.toarray()
    X_dense[X_dense.nonzero()] = 1.0
    reproduced_user_vector = uvec.dot(ivec.T)
    np.testing.assert_allclose(reproduced_user_vector, X_dense, rtol=1e-2, atol=1e-2)

    X_reproduced = rec.get_score_cold_user(X)
    np.testing.assert_allclose(X_reproduced, X_dense, rtol=1e-2, atol=1e-2)

    with pytest.raises(ValueError):
        _ = rec.compute_item_embedding(X.T)

    reproduced_item_vector = rec.compute_item_embedding(X)
    X_reproduced_item = rec.get_score_from_item_embedding(
        np.arange(X.shape[0]), reproduced_item_vector
    )
    np.testing.assert_allclose(X_reproduced_item, X_dense, rtol=1e-2, atol=1e-2)


@pytest.mark.xfail
def test_ials_cg_underfit(test_interaction_data: Dict[str, sps.csr_matrix]) -> None:
    X = test_interaction_data["X_small"]
    rec = IALSRecommender(
        X,
        n_components=4,
        alpha=0,
        reg=1e-2,
        use_cg=True,
        max_cg_steps=1,
        max_epoch=10,
    )
    with pytest.raises(RuntimeError):
        _ = rec.core_trainer.user
    rec.learn()
    assert rec.trainer is not None
    uvec = rec.core_trainer.user
    ivec = rec.core_trainer.item
    X_dense = X.toarray()
    reprod = uvec.dot(ivec.T)
    np.testing.assert_allclose(reprod, X_dense, rtol=1e-2, atol=1e-2)


def test_ials_overfit_nonzero_alpha(
    test_interaction_data: Dict[str, sps.csr_matrix]
) -> None:
    ALPHA = 4.5
    REG = 3
    X = test_interaction_data["X_small"]
    rec_chol = IALSRecommender(
        X, n_components=4, alpha=ALPHA, reg=REG, use_cg=False, max_epoch=5
    )
    rec_chol.learn()
    assert rec_chol.trainer is not None
    uvec_chol = rec_chol.trainer.core_trainer.transform_user(
        X.tocsr().astype(np.float32)
    )
    ivec_chol = rec_chol.trainer.core_trainer.transform_item(
        X.tocsr().astype(np.float32)
    )

    rec_cg = IALSRecommender(
        X,
        n_components=4,
        alpha=ALPHA,
        reg=REG,
        use_cg=True,
        max_cg_steps=4,
        max_epoch=5,
    )
    rec_cg.learn()
    assert rec_cg.trainer is not None
    uvec_cg = rec_cg.trainer.core_trainer.transform_user(X.tocsr())
    ivec_cg = rec_cg.trainer.core_trainer.transform_item(X.tocsr())

    np.testing.assert_allclose(uvec_chol, uvec_cg, atol=1e-3, rtol=1e-4)
    np.testing.assert_allclose(ivec_chol, ivec_cg, atol=1e-3, rtol=1e-4)


def ials_grad(
    X: sps.csr_matrix,
    u: np.ndarray,
    v: np.ndarray,
    reg: float,
    alpha: float,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    nu = u.shape[0]
    ni = v.shape[0]

    uv = u.dot(v.T)
    result_u = np.zeros_like(u)
    result_v = np.zeros_like(v)
    for uind in range(nu):
        for iind in range(ni):
            x = X[uind, iind]
            if x == 0:
                sc = uv[uind, iind]
            else:
                sc = (1 + alpha * np.log(1 + x / epsilon)) * (uv[uind, iind] - 1)

            result_u[uind, :] += v[iind] * sc
            result_v[iind, :] += u[uind] * sc
    result_u += reg * u
    result_v += reg * v
    return result_u, result_v


def test_ials_overfit_cholesky_logscale(
    test_interaction_data: Dict[str, sps.csr_matrix]
) -> None:

    ALPHA = 1.0
    REG = 0.1
    EPSILON = 1.0
    N_COMPONENTS = 1
    X = test_interaction_data["X_small"]
    rec_chol = IALSRecommender(
        X,
        n_components=N_COMPONENTS,
        alpha=ALPHA,
        reg=REG,
        use_cg=False,
        epsilon=EPSILON,
        confidence_scaling="log",
        max_epoch=200,
        init_std=1e-1,
    )
    rec_chol.learn()
    assert rec_chol.trainer is not None
    uvec_chol = rec_chol.get_user_embedding()
    ivec_chol_cold = rec_chol.compute_item_embedding(X)
    print(X.shape)

    grad_uvec_chol, grad_ivec_chol = ials_grad(
        X, uvec_chol, ivec_chol_cold, REG, ALPHA, EPSILON
    )

    np.testing.assert_allclose(grad_ivec_chol, np.zeros_like(grad_ivec_chol), atol=1e-5)
    np.testing.assert_allclose(grad_uvec_chol, np.zeros_like(grad_uvec_chol), atol=1e-5)
