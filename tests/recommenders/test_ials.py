from typing import Dict

import numpy as np
import pytest
import scipy.sparse as sps

from irspack.recommenders import IALSRecommender

X_small = sps.csr_matrix(
    np.asfarray([[6, 1, 2, 3, 4], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]])
)


def test_ials_overfit_cholesky(
    test_interaction_data: Dict[str, sps.csr_matrix]
) -> None:
    X = test_interaction_data["X_small"]
    rec = IALSRecommender(X, n_components=4, alpha=0, reg=0, use_cg=False)
    rec.learn()
    assert rec.trainer is not None
    uvec = rec.trainer.core_trainer.transform_user(X.tocsr().astype(np.float32))
    ivec = rec.trainer.core_trainer.transform_item(X.tocsr().astype(np.float32))
    X = X.toarray()
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
