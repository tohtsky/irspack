from collections import defaultdict

import numpy as np
import pytest
import scipy.sparse as sps
from sklearn.metrics import average_precision_score, ndcg_score

from irspack.evaluator import Evaluator
from irspack.recommenders.base import BaseRecommender


class MockRecommender(BaseRecommender):
    def __init__(self, X_all: sps.csr_matrix, scores: np.ndarray) -> None:
        super().__init__(X_all)
        self.scores = scores

    def get_score(self, user_indices: np.ndarray) -> np.ndarray:
        return self.scores[user_indices]

    def _learn(self) -> None:
        pass


@pytest.mark.parametrize("U, I", [(10, 5), (10, 30)])
def test_metrics(U: int, I: int) -> None:
    rns = np.random.RandomState(42)
    scores = rns.randn(U, I)
    X_gt = (rns.rand(U, I) >= 0.3).astype(np.float64)
    eval = Evaluator(sps.csr_matrix(X_gt), offset=0, cutoff=I, n_thread=4)
    # empty mask
    mock_rec = MockRecommender(sps.csr_matrix(X_gt.shape), scores)
    my_score = eval.get_score(mock_rec)
    sklearn_metrics = defaultdict(list)
    for i in range(scores.shape[0]):
        if X_gt[i].sum() == 0:
            continue
        sklearn_metrics["map"].append(average_precision_score(X_gt[i], scores[i]))
        sklearn_metrics["ndcg"].append(ndcg_score(X_gt[i][None, :], scores[i][None, :]))

    for key in ["map", "ndcg"]:
        assert my_score[key] == pytest.approx(np.mean(sklearn_metrics[key]), abs=1e-8)


@pytest.mark.parametrize("U, I, C", [(10, 5, 5), (10, 30, 29)])
def test_metrics_with_cutoff(U: int, I: int, C: int) -> None:
    rns = np.random.RandomState(42)
    scores = rns.randn(U, I)
    X_gt = (rns.rand(U, I) >= 0.3).astype(np.float64)
    eval = Evaluator(sps.csr_matrix(X_gt), offset=0, cutoff=C, n_thread=1)
    # empty mask
    mock_rec = MockRecommender(sps.csr_matrix(X_gt.shape), scores)
    my_score = eval.get_score(mock_rec)
    sklearn_ndcg = ndcg_score(X_gt, scores, k=C)
    assert my_score["ndcg"] == pytest.approx(sklearn_ndcg, abs=1e-8)
