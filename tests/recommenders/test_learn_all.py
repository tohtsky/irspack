import pickle
from inspect import isabstract

import numpy as np
import pytest
import scipy.sparse as sps

from irspack import BaseRecommender, Evaluator, get_recommender_class
from irspack.recommenders.base import RecommenderMeta

X_train = sps.csr_matrix(
    np.asfarray([[1, 1, 2, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
)

X_test = sps.csr_matrix(
    np.asfarray([[0, 0, 0, 1, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
)

rec_classes = list(RecommenderMeta.recommender_name_vs_recommender_class.keys())


@pytest.mark.parametrize("class_name", rec_classes)
def test_recs(class_name: str) -> None:
    """Test the learning of recommenders exit normally, and they are picklable.

    Args:
        class_name (str): The recommender class's name to be tested.
    """
    RecommenderClass = get_recommender_class(class_name)
    if isabstract(RecommenderClass):
        pytest.skip()
    rec = RecommenderClass.from_config(X_train, RecommenderClass.config_class())
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
