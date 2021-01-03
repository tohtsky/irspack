import numpy as np
import pytest
import scipy.sparse as sps

from irspack.definitions import InteractionMatrix, ProfileMatrix
from irspack.evaluator import Evaluator
from irspack.split import rowwise_train_test_split
from irspack.user_cold_start.evaluator import UserColdStartEvaluator
from irspack.user_cold_start.optimizers import LinearMethodOptimizer
from irspack.user_cold_start.recommenders import LinearMethodRecommender

RNS = np.random.RandomState(0)

profile = sps.csr_matrix(RNS.rand(3, 20) >= 0.7).astype(np.float64)
X_cf = sps.csr_matrix(RNS.rand(3, 10) >= 0.7).astype(np.float64)

profile = sps.vstack([profile for _ in range(100)])  # so many duplicates!
X_cf = sps.vstack([X_cf for _ in range(100)])


def test_lineamethod() -> None:
    config = LinearMethodOptimizer.split_and_optimize(
        X_cf, profile, evaluator_config=dict(cutoff=10), random_seed=0
    )
    rec = LinearMethodRecommender(X_cf, profile, **config).learn()
    evaluator = UserColdStartEvaluator(X_cf, profile, cutoff=5)
    scores = evaluator.get_score(rec)
    print(scores, config)
    assert scores["ndcg"] >= 0.8
