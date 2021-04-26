import time
from typing import Any, List

import numpy as np
from scipy import sparse as sps

from irspack import (
    BaseOptimizer,
    BaseRecommender,
    DenseScoreArray,
    InteractionMatrix,
    UserIndexArray,
)
from irspack.evaluator.evaluator import Evaluator
from irspack.parameter_tuning import Suggestion, UniformSuggestion
from irspack.parameter_tuning.autopilot import autopilot

TIMESCALE = 1.5

X_small = sps.csr_matrix(
    (np.random.RandomState(42).rand(100, 32) > 0.5).astype(np.float64)
)
X_answer = sps.csr_matrix(
    (np.random.RandomState(43).rand(*X_small.shape) > 0.5).astype(np.float64)
)


class AutopilotMockRecommender(BaseRecommender):
    def __init__(self, X: InteractionMatrix, wait_time: float = 1.0):
        super().__init__(X)
        self.wait_time = wait_time
        self.leaked = X_answer.toarray()

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        return self.leaked[user_indices]

    def _learn(self) -> None:
        time.sleep(self.wait_time)


class AutoPilotMockOptimizer(BaseOptimizer):
    recommender_class = AutopilotMockRecommender
    default_tune_range: List[Suggestion] = [UniformSuggestion("wait_time", 0.1, 4.0)]

    @classmethod
    def tune_range_given_memory_budget(
        cls, X: InteractionMatrix, memory_in_mb: int
    ) -> List[Suggestion]:
        return []


def test_autopilot() -> None:
    evaluator = Evaluator(X_answer, 0)
    algorithm_name, best_param, trial_df = autopilot(
        X_small,
        evaluator,
        memory_budget=1,
        n_trials=10,
        algorithms=["AutoPilotMock"],
        timeout_singlestep=2,
    )
    assert best_param["wait_time"] < 2.0
    wait_times = trial_df["AutoPilotMockOptimizer.wait_time"]
    assert np.all(trial_df.iloc[(wait_times.values > 2.0)]["ndcg@10"].isna())
    assert algorithm_name == "AutoPilotMockOptimizer"
