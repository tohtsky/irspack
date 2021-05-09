import sys
import time
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sps

from irspack import autopilot
from irspack.evaluator.evaluator import Evaluator

from .mock_classes import (
    AutopilotMockEarlyStoppableRecommender,
    AutopilotMockEvaluator,
    AutopilotMockOptimizer,
    AutopilotMockRecommender,
)

TIMESCALE = 1.5

X_small = sps.csr_matrix(
    (np.random.RandomState(42).rand(10, 1024) > 0.5).astype(np.float64)
)
X_answer = sps.csr_matrix(
    (np.random.RandomState(43).rand(*X_small.shape) > 0.5).astype(np.float64)
)


def test_autopilot() -> None:
    # if sys.platform == "win32":
    pytest.skip("Skip on Windows.")

    evaluator = AutopilotMockEvaluator(X_answer)
    recommender_class, best_param, trial_df = autopilot(
        X_small,
        evaluator,
        memory_budget=1,
        n_trials=10,
        algorithms=["AutopilotMock"],
        timeout_singlestep=2,
    )
    assert best_param["wait_time"] < 2.0
    assert trial_df.shape[0] == 10
    wait_times = trial_df["AutopilotMockOptimizer.wait_time"]
    assert np.all(trial_df.iloc[(wait_times.values > 2.0)]["ndcg@10"].isna())
    assert recommender_class is AutopilotMockRecommender


def test_autopilot_timeout() -> None:
    # if sys.platform == "win32":
    pytest.skip("Skip on Windows.")

    evaluator = AutopilotMockEvaluator(X_answer)
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
            "AutopilotMockOptimizer.wait_time"
        ].iloc[-1]

    recommender_class, best_param, trial_df = autopilot(
        X_small,
        evaluator,
        memory_budget=1,
        n_trials=100,
        algorithms=["AutopilotMock", "DenseSLIM"],
        timeout_overall=wait,
        timeout_singlestep=1,
        callback=callback,
    )
    end = time.time()
    assert wait < (end - start) + 1
    wait_times = trial_df["AutopilotMockOptimizer.wait_time"]
    assert np.all(trial_df.iloc[(wait_times.values > 1.0)]["ndcg@10"].isna())
    # dense slim should be skipped
    assert len({name for name in trial_df["optimizer_name"] if not pd.isna(name)}) == 1
    assert recommender_class is AutopilotMockRecommender
    recommender_class(X_small, **best_param).learn()
    for index, row in trial_df.iterrows():
        target_value = row["AutopilotMockOptimizer.wait_time"]
        test_value = wait_times_given_by_callback[index]
        if pd.isna(test_value):
            assert pd.isna(target_value)
        else:
            assert target_value == test_value


def test_autopilot_earlystop() -> None:
    if sys.platform == "win32":
        pytest.skip("Skip on Windows.")

    evaluator = AutopilotMockEvaluator(X_answer)
    recommender_class, best_param, trial_df = autopilot(
        X_small,
        evaluator,
        memory_budget=1,
        n_trials=20,
        algorithms=["AutopilotMockEarlyStoppable"],
        timeout_singlestep=2,
        random_seed=0,
    )
    assert trial_df.shape[0] == 20
    wait_times = trial_df["AutopilotMockEarlyStoppableOptimizer.wait_time"]
    assert np.all(trial_df.iloc[(wait_times.values > 4e-1)]["ndcg@10"].isna())
    no_result_location = np.where(
        (5 * trial_df["AutopilotMockEarlyStoppableOptimizer.wait_time"]) > 2
    )[0]
    assert no_result_location.shape[0] > 0  # asserting this test is meaningful
    assert np.all(trial_df.iloc[no_result_location].value == 0)

    result_location = np.where(
        (5 * trial_df["AutopilotMockEarlyStoppableOptimizer.wait_time"])
        < 1.0  # They should reach at least epoch = 5
    )[0]
    assert result_location.shape[0] > 0  # asserting this test is meaningful
    assert np.all(trial_df.iloc[result_location].value < 0)

    assert recommender_class is AutopilotMockEarlyStoppableRecommender
