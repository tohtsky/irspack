import pickle
from typing import List, Type

import numpy as np
import pytest
import scipy.sparse as sps

from irspack.evaluator import Evaluator
from irspack.recommenders import (
    AsymmetricCosineKNNRecommender,
    AsymmetricCosineUserKNNRecommender,
    BaseRecommender,
    CosineKNNRecommender,
    CosineUserKNNRecommender,
    DenseSLIMRecommender,
    IALSRecommender,
    JaccardKNNRecommender,
    NMFRecommender,
    P3alphaRecommender,
    RP3betaRecommender,
    SLIMRecommender,
    TopPopRecommender,
    TruncatedSVDRecommender,
    TverskyIndexKNNRecommender,
)

X_train = sps.csr_matrix(
    np.asfarray([[1, 1, 2, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
)

X_test = sps.csr_matrix(
    np.asfarray([[0, 0, 0, 1, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
)

rec_classes: List[Type[BaseRecommender]] = [
    TopPopRecommender,
    CosineKNNRecommender,
    AsymmetricCosineKNNRecommender,
    TverskyIndexKNNRecommender,
    JaccardKNNRecommender,
    CosineUserKNNRecommender,
    AsymmetricCosineUserKNNRecommender,
    P3alphaRecommender,
    RP3betaRecommender,
    TruncatedSVDRecommender,
    NMFRecommender,
    IALSRecommender,
    DenseSLIMRecommender,
    SLIMRecommender,
]
try:
    from irspack.recommenders.bpr import BPRFMRecommender

    rec_classes.append(BPRFMRecommender)
except:
    pass

try:
    from irspack.recommenders.multvae import MultVAERecommender

    rec_classes.append(MultVAERecommender)
except:
    pass


@pytest.mark.parametrize("RecommenderClass", rec_classes)
def test_recs(RecommenderClass: Type[BaseRecommender]) -> None:
    """Test the learning of recommenders exit normally, and they are picklable.

    Args:
        RecommenderClass (Type[BaseRecommender]): The recommender class to be tested.
    """
    rec = RecommenderClass(X_train)
    rec.learn()

    scores = rec.get_score(np.arange(X_train.shape[0]))
    eval = Evaluator(X_test, 0, 20)
    with pytest.raises(ValueError):
        eval.get_score(rec)
    metrics = eval.get_scores(rec, cutoffs=[X_train.shape[1]])
    assert np.all(np.isfinite(scores))
    assert np.all(~np.isnan(scores))
    for value in metrics.values():
        assert ~np.isnan(value)
        assert np.isfinite(value)
    with open("temp.pkl", "wb") as ofs:
        pickle.dump(rec, ofs)
    with open("temp.pkl", "rb") as ifs:
        rec_dumped: BaseRecommender = pickle.load(ifs)
    score_from_dumped = rec_dumped.get_score(np.arange(X_train.shape[0]))
    np.testing.assert_allclose(scores, score_from_dumped)
