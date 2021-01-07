import numpy as np
import pytest
import scipy.sparse as sps

from irspack.recommenders import TopPopRecommender

X = np.random.rand(200, 512)
X[X <= 0.9] = 0
X[X > 0.9] = 1
X = sps.csr_matrix(X)


def test_toppop() -> None:
    rec = TopPopRecommender(X)
    with pytest.raises(RuntimeError):
        _ = rec.get_score_cold_user_remove_seen(X)
    rec.learn()
    score = rec.get_score_cold_user(X)
    assert score.shape == X.shape
    np.testing.assert_allclose(score[0, :], score[-1, :])

    score_hot = rec.get_score_remove_seen(np.arange(X.shape[0]))
    assert np.all(np.isinf(score_hot[X.nonzero()]))
