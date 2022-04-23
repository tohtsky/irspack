import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pytest
import scipy.sparse as sps

from irspack.recommenders import IALSRecommender


def ials_grad(
    X: sps.csr_matrix,
    u: np.ndarray,
    v: np.ndarray,
    reg: float,
    alpha0: float,
    epsilon: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    weight: Callable[[float], float]
    if epsilon is None:
        weight = lambda x: x
    else:
        eps_copy = epsilon
        weight = lambda x: math.log(1 + x / eps_copy)
    nu = u.shape[0]
    ni = v.shape[0]

    uv = u.dot(v.T)
    result_u = np.zeros_like(u)
    result_v = np.zeros_like(v)
    for uind in range(nu):
        for iind in range(ni):
            x = X[uind, iind]
            if x == 0:
                sc = alpha0 * uv[uind, iind]
            else:
                sc = (alpha0 + weight(x)) * (uv[uind, iind] - 1)

            result_u[uind, :] += v[iind] * sc
            result_v[iind, :] += u[uind] * sc
    result_u += reg * u
    result_v += reg * v
    return result_u, result_v


def test_ials_overfit_cholesky(
    test_interaction_data: Dict[str, sps.csr_matrix]
) -> None:
    X = test_interaction_data["X_small"]
    rec = IALSRecommender(
        X,
        n_components=4,
        alpha0=100,
        reg=1e-1,
        solver_type="CHOLESKY",
        loss_type="ORIGINAL",
        max_epoch=100,
        n_threads=1,
        nu=0,
    )
    rec.learn()
    assert rec.trainer is not None
    uvec = rec.compute_user_embedding(X)
    ivec = rec.compute_item_embedding(X)
    X = X.toarray()
    X[X.nonzero()] = 1.0
    reprod = uvec.dot(ivec.T)
    np.testing.assert_allclose(reprod, X, rtol=1e-2, atol=1e-2)


def test_ials_loss_original(test_interaction_data: Dict[str, sps.csr_matrix]) -> None:
    X = test_interaction_data["X_small"]

    rec = IALSRecommender(
        X,
        n_components=2,
        alpha0=0.1,
        reg=1e-1,
        solver_type="CHOLESKY",
        loss_type="ORIGINAL",
        max_epoch=2,
        n_threads=1,
        nu=0,
    ).learn()
    assert rec.trainer is not None
    uvec = rec.get_user_embedding()
    ivec = rec.get_item_embedding()
    ui = uvec.dot(ivec.T)
    row, col = X.nonzero()

    # bruteforce computation of iALS loss
    loss_manual = (X.data + rec.alpha0).dot((ui[row, col] - 1) ** 2)
    ui[row, col] = 0.0
    loss_manual += rec.alpha0 * ui.ravel().dot(ui.ravel())
    loss_manual += rec.reg * ((uvec**2).sum() + (ivec**2).sum())
    loss_manual /= 2
    assert rec.trainer.compute_loss() == pytest.approx(loss_manual)


def test_ials_loss_ialspp(test_interaction_data: Dict[str, sps.csr_matrix]) -> None:
    X = test_interaction_data["X_small"]

    rec = IALSRecommender(
        X,
        n_components=2,
        alpha0=0.1,
        reg=1e-1,
        solver_type="CHOLESKY",
        loss_type="IALSPP",
        max_epoch=2,
        n_threads=1,
        nu=0,
    ).learn()
    assert rec.trainer is not None
    uvec = rec.get_user_embedding()
    ivec = rec.get_item_embedding()
    ui = uvec.dot(ivec.T)
    row, col = X.nonzero()

    # bruteforce computation of iALS loss
    loss_manual = (X.data).dot((ui[row, col] - 1) ** 2)
    loss_manual += rec.alpha0 * ui.ravel().dot(ui.ravel())
    loss_manual += rec.reg * ((uvec**2).sum() + (ivec**2).sum())
    loss_manual /= 2
    assert rec.trainer.compute_loss() == pytest.approx(loss_manual)


def test_ials_overfit_cg(test_interaction_data: Dict[str, sps.csr_matrix]) -> None:
    X = test_interaction_data["X_small"]
    rec = IALSRecommender(
        X,
        n_components=3,
        alpha0=100,
        loss_type="ORIGINAL",
        reg=1e-1,
        solver_type="CG",
        max_cg_steps=3,
        max_epoch=100,
        nu=0,
    )
    rec.learn()
    assert rec.trainer is not None
    uvec = rec.compute_user_embedding(X.tocsr().astype(np.float32))
    ivec = rec.compute_item_embedding(X.tocsr().astype(np.float32))
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


@pytest.mark.parametrize(["subspace_dimension"], [(1,), (2,), (3,), (4,)])
def test_ials_overfit_ialspp(
    subspace_dimension: int, test_interaction_data: Dict[str, sps.csr_matrix]
) -> None:
    X = test_interaction_data["X_small"].copy()
    ALPHA0 = 100
    REG = 1.0
    rec = IALSRecommender(
        X,
        n_components=4,
        alpha0=ALPHA0,
        reg=REG,
        solver_type="IALSPP",
        loss_type="ORIGINAL",
        max_epoch=300,
        n_threads=1,
        ialspp_subspace_dimension=subspace_dimension,
        nu=0,
    )
    rec.learn()
    assert rec.trainer is not None
    uvec = rec.get_user_embedding()
    ivec = rec.get_item_embedding()
    X = X.toarray()
    X[X.nonzero()] = 1.0
    reprod = uvec.dot(ivec.T)
    np.testing.assert_allclose(reprod, X, rtol=1e-2, atol=1e-2)


@pytest.mark.xfail
def test_ials_cg_underfit(test_interaction_data: Dict[str, sps.csr_matrix]) -> None:
    X = test_interaction_data["X_small"]
    rec = IALSRecommender(
        X,
        n_components=4,
        alpha0=1e3,
        reg=1e-3,
        solver_type="CG",
        max_cg_steps=1,
        max_epoch=10,
        nu=0,
    )
    with pytest.raises(RuntimeError):
        _ = rec.trainer_as_ials.core_trainer.user
    rec.learn()
    assert rec.trainer is not None
    uvec = rec.trainer.core_trainer.user
    ivec = rec.trainer.core_trainer.item
    X_dense = X.toarray()
    X_dense[X_dense.nonzero()] = 1.0
    reprod = uvec.dot(ivec.T)
    np.testing.assert_allclose(reprod, X_dense, rtol=1e-2, atol=1e-2)


def test_ials_overfit_nonzero_alpha(
    test_interaction_data: Dict[str, sps.csr_matrix]
) -> None:
    ALPHA = 4.5
    REG = 3
    X = test_interaction_data["X_small"]
    rec_chol = IALSRecommender(
        X,
        n_components=4,
        alpha0=1 / ALPHA,
        reg=REG,
        solver_type="CHOLESKY",
        max_epoch=5,
    )
    rec_chol.learn()
    assert rec_chol.trainer is not None
    uvec_chol = rec_chol.compute_user_embedding(X.tocsr().astype(np.float32))
    ivec_chol = rec_chol.compute_item_embedding(X.tocsr().astype(np.float32))

    rec_cg = IALSRecommender(
        X,
        n_components=4,
        alpha0=1 / ALPHA,
        reg=REG,
        solver_type="CG",
        max_cg_steps=5,
        max_epoch=5,
    )
    rec_cg.learn()
    assert rec_cg.trainer is not None
    uvec_cg = rec_cg.compute_user_embedding(X.tocsr())
    ivec_cg = rec_cg.compute_item_embedding(X.tocsr())

    np.testing.assert_allclose(uvec_chol, uvec_cg, atol=1e-3, rtol=1e-4)
    np.testing.assert_allclose(ivec_chol, ivec_cg, atol=1e-3, rtol=1e-4)


def test_ials_overfit_cholesky_logscale(
    test_interaction_data: Dict[str, sps.csr_matrix]
) -> None:

    ALPHA0 = 2.4
    REG = 1.1
    EPSILON = 3.0
    N_COMPONENTS = 5
    X = test_interaction_data["X_small"]
    rec_chol = IALSRecommender(
        X,
        n_components=N_COMPONENTS,
        alpha0=ALPHA0,
        reg=REG,
        nu=0,
        nu_star=0,
        solver_type="CHOLESKY",
        loss_type="ORIGINAL",
        epsilon=EPSILON,
        confidence_scaling="log",
        max_epoch=200,
        init_std=1e-1,
    )
    rec_chol.learn()
    assert rec_chol.trainer is not None
    uvec_chol = rec_chol.get_user_embedding()
    ivec_chol_cold = rec_chol.compute_item_embedding(X)

    grad_uvec_chol, grad_ivec_chol = ials_grad(
        X, uvec_chol, ivec_chol_cold, REG, ALPHA0, EPSILON
    )

    np.testing.assert_allclose(grad_ivec_chol, np.zeros_like(grad_ivec_chol), atol=1e-5)
    np.testing.assert_allclose(grad_uvec_chol, np.zeros_like(grad_uvec_chol), atol=1e-5)
