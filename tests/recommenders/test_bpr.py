from typing import Dict

import numpy as np
import pytest
import scipy.sparse as sps


def test_bprFM(test_interaction_data: Dict[str, sps.csr_matrix]) -> None:

    try:
        from irspack.recommenders import BPRFMRecommender
    except:
        pytest.skip("lightfm not found.")
        raise
    X = test_interaction_data["X_small"]
    rec = BPRFMRecommender(X, n_components=min(X.shape), max_epoch=30)
    with pytest.raises(RuntimeError):
        _ = rec.fm.user_embeddings
    rec.learn()
    score = rec.get_user_embedding().dot(rec.get_item_embedding().T)
    assert score.shape == X.shape
    score_2 = rec.get_score_remove_seen_block(0, X.shape[0])
    assert np.all(np.isneginf(score_2[X.nonzero()]))
