import numpy as np
import pytest
import scipy.sparse as sps

RNS = np.random.RandomState(0)

X_dense = RNS.rand(200, 512)
X_dense[X_dense <= 0.9] = 0
X_dense[X_dense > 0.9] = 1
X: sps.csr_matrix = sps.csr_matrix(X_dense)


def test_truncsvd() -> None:
    try:
        from irspack.recommenders import TruncatedSVDRecommender
    except ImportError:
        pytest.skip("Trunc svd not found.")

    with pytest.warns(UserWarning):
        overfit_rec = TruncatedSVDRecommender(X, n_components=X.shape[1] + 1)

    with pytest.raises(RuntimeError):
        overfit_rec.get_score_block(0, X.shape[0])

    with pytest.raises(RuntimeError):
        overfit_rec.get_item_embedding()

    with pytest.raises(RuntimeError):
        overfit_rec.get_user_embedding()

    overfit_rec.learn()
    X_reproduced = overfit_rec.get_score_block(0, X.shape[0])
    X_frobenius = (X.data**2).sum()
    residual = (X_reproduced - X).A1
    assert ((residual**2).sum() / X_frobenius) < 1e-3

    X_reproduced_cold = overfit_rec.get_score_cold_user(X)
    residual_cold = (X_reproduced_cold - X).A1
    assert ((residual_cold**2).sum() / X_frobenius) < 1e-3

    score_user_to_item = overfit_rec.get_score_from_user_embedding(
        overfit_rec.get_user_embedding()
    )
    assert score_user_to_item.shape == X.shape
    score_item_to_user = overfit_rec.get_score_from_item_embedding(
        np.arange(X.shape[0]), overfit_rec.get_item_embedding()
    )
    assert score_item_to_user.shape == X.shape
