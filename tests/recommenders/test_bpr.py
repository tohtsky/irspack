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
    rec = BPRFMRecommender(X, n_components=min(X.shape), train_epochs=30)
    with pytest.raises(RuntimeError):
        _ = rec.fm.user_embeddings
    rec.learn()
    score = rec.get_user_embedding().dot(rec.get_item_embedding().T)
    assert score.shape == X.shape
    score_2 = rec.get_score_remove_seen_block(0, X.shape[0])
    assert np.all(np.isneginf(score_2[X.nonzero()]))

    score_user_to_item = rec.get_score_from_user_embedding(rec.get_user_embedding())
    assert score_user_to_item.shape == X.shape
    score_item_to_user = rec.get_score_from_item_embedding(
        np.arange(X.shape[0]), rec.get_item_embedding()
    )
    assert score_item_to_user.shape == X.shape
