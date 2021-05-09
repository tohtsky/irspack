import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sps

from irspack import (
    BaseOptimizer,
    BaseRecommender,
    DenseScoreArray,
    InteractionMatrix,
    UserIndexArray,
    autopilot,
)
from irspack.evaluator.evaluator import Evaluator
from irspack.parameter_tuning import Suggestion, UniformSuggestion

SKIP_TEST = os.environ.get("IRSPACK_TESTING", "false") != "true"

TIMESCALE = 1.5

X_small = sps.csr_matrix(
    (np.random.RandomState(42).rand(10, 1024) > 0.5).astype(np.float64)
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


@pytest.mark.skipif(SKIP_TEST, reason="May not work on non-Linux.")
def test_autopilot() -> None:
    if sys.platform == "win32":
        pytest.skip("Skip on Windows.")

    evaluator = Evaluator(X_answer, 0)
    recommender_class, best_param, trial_df = autopilot(
        X_small,
        evaluator,
        memory_budget=1,
        n_trials=10,
        algorithms=["AutoPilotMock"],
        timeout_singlestep=2,
    )
    assert best_param["wait_time"] < 2.0
    assert trial_df.shape[0] == 10
    wait_times = trial_df["AutoPilotMockOptimizer.wait_time"]
    assert np.all(trial_df.iloc[(wait_times.values > 2.0)]["ndcg@10"].isna())
    assert recommender_class is AutopilotMockRecommender


@pytest.mark.skipif(SKIP_TEST, reason="May not work on non-Linux.")
def test_autopilot_timeout() -> None:
    if sys.platform == "win32":
        pytest.skip("Skip on Windows.")

    evaluator = Evaluator(X_answer, 0)
    wait = 20
    with pytest.raises(RuntimeError):
        # no available algorithm
        autopilot(
            X_small,
            evaluator,
            memory_budget=1,
            n_trials=100,
            algorithms=["DenseSLIM"],
            timeout_overall=5,
            timeout_singlestep=1,
        )

    start = time.time()

    wait_times_given_by_callback: Dict[int, float] = {}

    def callback(trial_number: int, history_df: pd.DataFrame) -> None:
        wait_times_given_by_callback[trial_number] = history_df[
            "AutoPilotMockOptimizer.wait_time"
        ].iloc[-1]

    recommender_class, best_param, trial_df = autopilot(
        X_small,
        evaluator,
        memory_budget=1,
        n_trials=100,
        algorithms=["AutoPilotMock", "DenseSLIM"],
        timeout_overall=wait,
        timeout_singlestep=1,
        callback=callback,
    )
    end = time.time()
    assert wait < (end - start) + 1
    wait_times = trial_df["AutoPilotMockOptimizer.wait_time"]
    assert np.all(trial_df.iloc[(wait_times.values > 1.0)]["ndcg@10"].isna())
    # dense slim should be skipped
    assert len({name for name in trial_df["optimizer_name"] if not pd.isna(name)}) == 1
    assert recommender_class is AutopilotMockRecommender
    recommender_class(X_small, **best_param).learn()
    for index, row in trial_df.iterrows():
        target_value = row["AutoPilotMockOptimizer.wait_time"]
        test_value = wait_times_given_by_callback[index]
        if pd.isna(test_value):
            assert pd.isna(target_value)
        else:
            assert target_value == test_value
