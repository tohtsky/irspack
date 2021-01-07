import numpy as np
import pytest
import scipy.sparse as sps

from irspack.definitions import InteractionMatrix, ProfileMatrix
from irspack.evaluator import Evaluator
from irspack.split import rowwise_train_test_split

RNS = np.random.RandomState(0)

profile = sps.csr_matrix(RNS.rand(3, 10) >= 0.7).astype(np.float64)
X_cf = sps.csr_matrix(RNS.rand(3, 30) >= 0.7).astype(np.float64)

profile = sps.vstack([profile for _ in range(10)])  # so many duplicates!
X_cf = sps.vstack([X_cf for _ in range(10)])


@pytest.mark.parametrize("X, profile", [(X_cf, profile)])
def test_cb2cf(X: InteractionMatrix, profile: ProfileMatrix) -> None:

    """Fit IALS & let mlp overfit.

    Args:
        X (InteractionMatrix): user_item interaction matrix
        profile (ProfileMatrix): profile
    """
    try:
        from irspack.user_cold_start.cb2cf import CB2IALSOptimizer
    except:
        pytest.skip("Failed to import jax.")
        raise

    X_cf_train_all, X_val = rowwise_train_test_split(
        X_cf, test_ratio=0.5, random_seed=0
    )
    evaluator = Evaluator(X_val, 0)
    optim = CB2IALSOptimizer(
        X_cf_train_all,
        evaluator,
        profile,
    )
    cb2cfrec, t, mlp_config = optim.search_all(
        20,
        cf_fixed_params=dict(n_components=5, alpha=0, reg=1e-3, max_cg_steps=30),
        random_seed=0,
    )
    vec_reconstruction = cb2cfrec.mlp.predict(profile.astype(np.float32).toarray())
    vec_target = cb2cfrec.cf_rec.get_user_embedding()

    residual = ((vec_reconstruction - vec_target) ** 2).sum() / (vec_target ** 2).sum()
    assert residual <= 1e-1
