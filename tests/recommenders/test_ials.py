import math
import pickle
from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np
import pytest
import scipy.sparse as sps

from irspack.evaluation import Evaluator
from irspack.recommenders._ials_core import (
    IALSModelConfigBuilder,
    IALSSolverConfigBuilder,
    IALSTrainer,
)
from irspack.recommenders.ials import IALSRecommender
from irspack.utils import rowwise_train_test_split


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
    test_interaction_data: Dict[str, sps.csr_matrix],
) -> None:
    X = test_interaction_data["X_small"]
    rec = IALSRecommender(
        X,
        n_components=4,
        alpha0=100,
        reg=1e-1,
        solver_type="CHOLESKY",
        loss_type="ORIGINAL",
        train_epochs=100,
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


@pytest.mark.parametrize(
    ("solver_type", "max_cg_steps"),
    [("CHOLESKY", 3), ("CG", 0)],
)
@pytest.mark.parametrize("feature_type", ["dense", "sparse"])
@pytest.mark.parametrize("n_threads", [1, 3])
def test_feature_aware_ials_weighted_updates_objective_and_local_stability(
    solver_type: Literal["CG", "CHOLESKY"],
    max_cg_steps: int,
    feature_type: str,
    n_threads: int,
) -> None:
    interaction_dense = np.array(
        [[1, 0, 2, 1], [0, 3, 0, 0], [1, 1, 0, 4]], dtype=np.float64
    )
    interaction = sps.csr_matrix(interaction_dense.astype(np.float32))
    user_feature_dense = np.array([[1, 0.2], [0.3, 1], [0.7, -0.2]], dtype=np.float32)
    item_feature_dense = np.array(
        [[1, 0, 0.1], [0, 1, 0.2], [0.5, 0.2, 1], [-0.2, 0.8, 0.4]],
        dtype=np.float32,
    )
    if feature_type == "dense":
        user_features = user_feature_dense
        item_features = item_feature_dense
    else:
        user_features = sps.csr_matrix(user_feature_dense)
        item_features = sps.csr_matrix(item_feature_dense)
    alpha0 = 0.7
    reg = 0.03
    nu = 0.6
    lambda_user_feature = 0.11
    lambda_item_feature = 0.17

    rec = IALSRecommender(
        interaction,
        n_components=3,
        alpha0=alpha0,
        reg=reg,
        nu=nu,
        solver_type=solver_type,
        max_cg_steps=max_cg_steps,
        prediction_time_max_cg_steps=0,
        loss_type="ORIGINAL",
        user_features=user_features,
        item_features=item_features,
        lambda_user_feature=lambda_user_feature,
        lambda_item_feature=lambda_item_feature,
        train_epochs=500,
        n_threads=n_threads,
        random_seed=0,
    ).learn()
    core = rec.trainer_as_ials.core_trainer
    user = np.asarray(core.user, dtype=np.float64)
    item = np.asarray(core.item, dtype=np.float64)
    user_weight = np.asarray(core.user_feature_weight, dtype=np.float64)
    item_weight = np.asarray(core.item_feature_weight, dtype=np.float64)
    user_nnz = np.count_nonzero(interaction_dense, axis=1)
    item_nnz = np.count_nonzero(interaction_dense, axis=0)
    user_reg = reg * (alpha0 * interaction.shape[1] + user_nnz) ** nu
    item_reg = reg * (alpha0 * interaction.shape[0] + item_nnz) ** nu

    expected_user_weight = np.linalg.solve(
        user_feature_dense.T @ (user_reg[:, None] * user_feature_dense)
        + lambda_user_feature * np.eye(user_feature_dense.shape[1]),
        user_feature_dense.T @ (user_reg[:, None] * user),
    )
    expected_item_weight = np.linalg.solve(
        item_feature_dense.T @ (item_reg[:, None] * item_feature_dense)
        + lambda_item_feature * np.eye(item_feature_dense.shape[1]),
        item_feature_dense.T @ (item_reg[:, None] * item),
    )
    np.testing.assert_allclose(user_weight, expected_user_weight, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(item_weight, expected_item_weight, rtol=2e-6, atol=2e-6)

    def objective(values: list[np.ndarray]) -> float:
        user_value, item_value, user_weight_value, item_weight_value = values
        score = user_value @ item_value.T
        observed = interaction_dense.astype(bool)
        interaction_loss = alpha0 * np.square(score[~observed]).sum()
        interaction_loss += np.sum(
            (interaction_dense[observed] + alpha0) * np.square(score[observed] - 1)
        )
        residual_regularization = np.sum(
            user_reg[:, None]
            * np.square(user_value - user_feature_dense @ user_weight_value)
        ) + np.sum(
            item_reg[:, None]
            * np.square(item_value - item_feature_dense @ item_weight_value)
        )
        feature_regularization = (
            lambda_user_feature * np.square(user_weight_value).sum()
            + lambda_item_feature * np.square(item_weight_value).sum()
        )
        return float(
            (interaction_loss + residual_regularization + feature_regularization) / 2
        )

    parameters = [user, item, user_weight, item_weight]
    optimum = objective(parameters)
    np.testing.assert_allclose(
        core.compute_loss(rec.trainer_as_ials.solver_config),
        optimum,
        rtol=2e-6,
        atol=2e-6,
    )

    def solve_embeddings(
        histories: np.ndarray,
        other_factor: np.ndarray,
        prior: np.ndarray,
        regularization: np.ndarray,
    ) -> np.ndarray:
        result = []
        base_gram = alpha0 * other_factor.T @ other_factor
        for row, prior_row, row_reg in zip(histories, prior, regularization):
            lhs = base_gram + row_reg * np.eye(other_factor.shape[1])
            rhs = row_reg * prior_row
            for other_index, value in enumerate(row):
                if value:
                    factor = other_factor[other_index]
                    lhs += value * np.outer(factor, factor)
                    rhs += (alpha0 + value) * factor
            result.append(np.linalg.solve(lhs, rhs))
        return np.asarray(result)

    expected_user = solve_embeddings(
        interaction_dense,
        item,
        user_feature_dense @ user_weight,
        user_reg,
    )
    expected_item = solve_embeddings(
        interaction_dense.T,
        user,
        item_feature_dense @ item_weight,
        item_reg,
    )
    np.testing.assert_allclose(
        rec.compute_user_embedding(interaction, user_features=user_features),
        expected_user,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        rec.compute_item_embedding(interaction, item_features=item_features),
        expected_item,
        rtol=2e-5,
        atol=2e-5,
    )

    # This is deliberately a black-box numerical stability check in addition
    # to the exact block-update checks above. Testing both signs makes every
    # sampled line through the joint parameter space bidirectional, while the
    # small radius exposes first-order descent that a larger perturbation can
    # hide behind positive curvature.
    rng = np.random.default_rng(1)
    for radius in (1e-5, 1e-3):
        for _ in range(128):
            direction = [rng.standard_normal(value.shape) for value in parameters]
            direction_norm = np.sqrt(sum(np.square(value).sum() for value in direction))
            direction = [value / direction_norm for value in direction]
            for sign in (-1, 1):
                perturbed = [
                    value + sign * radius * delta
                    for value, delta in zip(parameters, direction)
                ]
                assert objective(perturbed) >= optimum - 5e-10


def test_feature_aware_ials_feature_only_api_and_core_pickle() -> None:
    interaction = sps.csr_matrix(
        np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.float32)
    )
    user_features = sps.csr_matrix(np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float32))
    item_features = sps.csr_matrix(
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    )

    rec = IALSRecommender(
        interaction,
        n_components=2,
        alpha0=10.0,
        reg=1e-3,
        nu=0,
        solver_type="CG",
        max_cg_steps=0,
        loss_type="ORIGINAL",
        user_features=user_features,
        item_features=item_features,
        lambda_user_feature=1e-3,
        lambda_item_feature=1e-3,
        train_epochs=4,
        n_threads=1,
        random_seed=0,
    ).learn()

    core = rec.trainer_as_ials.core_trainer
    user_prior = user_features @ np.asarray(core.user_feature_weight)
    item_prior = item_features @ np.asarray(core.item_feature_weight)
    empty_user_history = sps.csr_matrix((user_features.shape[0], interaction.shape[1]))
    empty_item_history = sps.csr_matrix((interaction.shape[0], item_features.shape[0]))
    expected_user_embedding = rec.compute_user_embedding(
        empty_user_history, user_features=user_features
    )
    expected_item_embedding = rec.compute_item_embedding(
        empty_item_history, item_features=item_features
    )

    np.testing.assert_allclose(
        rec.compute_user_embedding_from_features(user_features),
        expected_user_embedding,
    )
    np.testing.assert_allclose(
        rec.compute_item_embedding_from_features(item_features),
        expected_item_embedding,
    )
    np.testing.assert_allclose(
        rec.get_score_cold_user_from_features(user_features),
        expected_user_embedding @ rec.get_item_embedding().T,
    )
    np.testing.assert_allclose(
        rec.get_score_from_item_features(
            np.arange(interaction.shape[0]), item_features
        ),
        rec.get_user_embedding() @ expected_item_embedding.T,
    )

    core_dumped = pickle.loads(pickle.dumps(core))
    np.testing.assert_allclose(
        np.asarray(core_dumped.user_feature_weight),
        np.asarray(core.user_feature_weight),
    )
    np.testing.assert_allclose(
        np.asarray(core_dumped.item_feature_weight),
        np.asarray(core.item_feature_weight),
    )
    np.testing.assert_allclose(
        core_dumped.transform_user_feature(user_features),
        user_prior,
    )
    np.testing.assert_allclose(
        core_dumped.transform_item_feature(item_features),
        item_prior,
    )


@pytest.mark.parametrize("solver_type", ["CHOLESKY", "CG"])
def test_feature_only_embedding_rejects_singular_empty_history(
    solver_type: Literal["CG", "CHOLESKY"],
) -> None:
    interaction = sps.csr_matrix(
        np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.float32)
    )
    features = np.eye(3, dtype=np.float32)
    rec = IALSRecommender(
        interaction,
        n_components=2,
        alpha0=0.0,
        reg=0.1,
        nu=1.0,
        solver_type=solver_type,
        max_cg_steps=0,
        loss_type="ORIGINAL",
        user_features=features,
        item_features=features,
        lambda_user_feature=0.1,
        lambda_item_feature=0.1,
        train_epochs=2,
        n_threads=1,
        random_seed=0,
    ).learn()

    message = "not uniquely defined for an empty interaction row"
    with pytest.raises(ValueError, match=message):
        rec.compute_user_embedding_from_features(features)
    with pytest.raises(ValueError, match=message):
        rec.compute_item_embedding_from_features(features)


def test_feature_aware_ials_dense_features_and_hybrid_transform() -> None:
    interaction_dense = np.array([[1, 0, 1], [0, 2, 1], [1, 1, 0]], dtype=np.float32)
    interaction = sps.csr_matrix(interaction_dense)
    user_features = np.array([[0.1, 0.2], [0.3, -0.1], [0.0, 0.4]], dtype=np.float32)
    item_features = np.array(
        [[0.5, 0.0, 0.1], [0.2, -0.3, 0.0], [0.0, 0.1, 0.4]],
        dtype=np.float32,
    )
    alpha0 = 0.5
    reg = 1e-3
    rec = IALSRecommender(
        interaction,
        n_components=2,
        alpha0=alpha0,
        reg=reg,
        nu=0,
        solver_type="CHOLESKY",
        loss_type="ORIGINAL",
        user_features=user_features,
        item_features=item_features,
        lambda_user_feature=1e-3,
        lambda_item_feature=1e-3,
        train_epochs=3,
        n_threads=1,
        random_seed=0,
    ).learn()
    core = rec.trainer_as_ials.core_trainer

    expected_user_prior = user_features @ np.asarray(core.user_feature_weight)
    expected_item_prior = item_features @ np.asarray(core.item_feature_weight)
    empty_user_history = sps.csr_matrix((user_features.shape[0], interaction.shape[1]))
    empty_item_history = sps.csr_matrix((interaction.shape[0], item_features.shape[0]))
    np.testing.assert_allclose(
        rec.compute_user_embedding_from_features(user_features),
        rec.compute_user_embedding(empty_user_history, user_features=user_features),
    )
    np.testing.assert_allclose(
        rec.compute_item_embedding_from_features(item_features),
        rec.compute_item_embedding(empty_item_history, item_features=item_features),
    )

    item = rec.get_item_embedding()
    cold_user_lhs = alpha0 * item.T @ item + reg * np.eye(item.shape[1])
    expected_cold_users = np.linalg.solve(
        cold_user_lhs, (reg * expected_user_prior).T
    ).T
    np.testing.assert_allclose(
        rec.compute_user_embedding_from_features(user_features),
        expected_cold_users,
        rtol=1e-5,
        atol=1e-6,
    )

    hybrid_user = rec.compute_user_embedding(
        interaction,
        user_features=user_features,
    )
    expected_hybrid_user = np.zeros_like(hybrid_user)
    base_gram = alpha0 * item.T @ item
    eye = np.eye(item.shape[1], dtype=np.float32)
    for row_index in range(interaction.shape[0]):
        row = interaction.getrow(row_index)
        nnz = row.nnz
        lhs = base_gram + reg * eye
        rhs = reg * expected_user_prior[row_index]
        for item_index, value in zip(row.indices, row.data):
            y = item[item_index]
            lhs += value * np.outer(y, y)
            rhs += (alpha0 + value) * y
        expected_hybrid_user[row_index] = np.linalg.solve(lhs, rhs)

    np.testing.assert_allclose(hybrid_user, expected_hybrid_user, rtol=1e-5)
    np.testing.assert_allclose(
        rec.get_score_cold_user(interaction, user_features=user_features),
        hybrid_user @ item.T,
    )


def test_ials_loss_original(test_interaction_data: Dict[str, sps.csr_matrix]) -> None:
    X = test_interaction_data["X_small"]

    rec = IALSRecommender(
        X,
        n_components=2,
        alpha0=0.1,
        reg=1e-1,
        solver_type="CHOLESKY",
        loss_type="ORIGINAL",
        train_epochs=2,
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


@pytest.mark.parametrize("alpha0", [0.0, 0.1])
def test_ials_loss_ialspp(
    test_interaction_data: Dict[str, sps.csr_matrix], alpha0: float
) -> None:
    X = test_interaction_data["X_small"]

    rec = IALSRecommender(
        X,
        n_components=2,
        alpha0=alpha0,
        reg=1e-1,
        solver_type="CHOLESKY",
        loss_type="IALSPP",
        train_epochs=2,
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
    loss_manual += alpha0 * ui.ravel().dot(ui.ravel())
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
        train_epochs=100,
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


@pytest.mark.parametrize("n_threads", [1, 4, 64])
def test_user_scores_batching(n_threads: int) -> None:
    rng = np.random.default_rng(0)
    n_users, n_items, n_components = 513, 257, 31
    interaction = sps.csr_matrix((n_users, n_items), dtype=np.float32)
    model_config = IALSModelConfigBuilder().set_K(n_components).build()
    solver_config = IALSSolverConfigBuilder().set_n_threads(n_threads).build()
    trainer = IALSTrainer(model_config, interaction)
    user = rng.standard_normal((n_users, n_components)).astype(np.float32)
    item = rng.standard_normal((n_items, n_components)).astype(np.float32)
    trainer.user = user
    trainer.item = item

    for begin, end in [(0, n_users), (17, 193), (n_users, n_users)]:
        np.testing.assert_allclose(
            trainer.user_scores(begin, end, solver_config),
            user[begin:end] @ item.T,
            rtol=2e-5,
            atol=2e-5,
        )


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
        train_epochs=300,
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
        train_epochs=10,
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
    test_interaction_data: Dict[str, sps.csr_matrix],
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
        train_epochs=5,
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
        train_epochs=5,
    )
    rec_cg.learn()
    assert rec_cg.trainer is not None
    uvec_cg = rec_cg.compute_user_embedding(X.tocsr())
    ivec_cg = rec_cg.compute_item_embedding(X.tocsr())

    np.testing.assert_allclose(uvec_chol, uvec_cg, atol=1e-3, rtol=1e-4)
    np.testing.assert_allclose(ivec_chol, ivec_cg, atol=1e-3, rtol=1e-4)


def test_ials_overfit_cholesky_logscale(
    test_interaction_data: Dict[str, sps.csr_matrix],
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
        train_epochs=200,
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


def test_ials_tuning_with_n_startup_trials(
    test_interaction_data: Dict[str, sps.csr_matrix],
) -> None:
    X = test_interaction_data["X_small"]
    X_tr, X_te = rowwise_train_test_split(X, random_state=0)
    cutoff = min(10, X.shape[1])
    bp, history = IALSRecommender.tune(
        X_tr,
        Evaluator(X_te, cutoff=cutoff),
        tuning_random_seed=0,
        prunning_n_startup_trials=20,
        n_trials=20,
    )
    assert history[f"ndcg@{cutoff}"].isna().sum() == 0
    assert "train_epochs" in bp
    assert bp["train_epochs"] <= 16


def test_ials_tuning_with_too_early_n_startup(
    test_interaction_data: Dict[str, sps.csr_matrix],
) -> None:
    X = test_interaction_data["X_small"]
    X_tr, X_te = rowwise_train_test_split(X, random_state=0)
    cutoff = min(10, X.shape[1])
    bp, history = IALSRecommender.tune(
        X_tr,
        Evaluator(X_te, cutoff=cutoff),
        tuning_random_seed=0,
        prunning_n_startup_trials=1,
        n_trials=20,
        max_epoch=3,
    )
    assert (~history[f"ndcg@{cutoff}"].isna()).sum() > 0
    assert "train_epochs" in bp
    assert bp["train_epochs"] <= 3
