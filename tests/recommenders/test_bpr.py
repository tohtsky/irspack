import numpy as np
import pytest
import scipy.sparse as sps

from irspack.recommenders import BPRFMRecommender

X_small = sps.csr_matrix(
    np.asfarray([[1, 1, 2, 3, 4], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
)


@pytest.mark.parametrize("X", [X_small])
def test_bprFM(X: sps.csr_matrix) -> None:
    rec = BPRFMRecommender(X_small, n_components=min(X.shape), max_epoch=30)
    with pytest.raises(RuntimeError):
        _ = rec.fm.user_embeddings
    rec.learn()
    score = rec.get_user_embedding().dot(rec.get_item_embedding().T)
    assert score.shape == X_small.shape
    score_2 = rec.get_score_remove_seen_block(0, X.shape[0])
    assert np.all(np.isneginf(score_2[X.nonzero()]))
