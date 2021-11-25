import os
import pickle
from contextlib import redirect_stderr, redirect_stdout
from logging import getLogger
from time import sleep
from typing import IO, Any, Dict

import numpy as np
import pytest
import scipy.sparse as sps

from irspack import (
    BaseOptimizerWithEarlyStopping,
    BaseRecommender,
    BaseRecommenderWithEarlyStopping,
    DenseScoreArray,
    Evaluator,
    InteractionMatrix,
    TargetMetric,
    UserIndexArray,
)
from irspack.parameter_tuning import UniformSuggestion
from irspack.recommenders.base_earlystop import TrainerBase

X_small = sps.csr_matrix(
    (np.random.RandomState(42).rand(100, 32) > 0.5).astype(np.float64)
)
X_large = sps.csr_matrix(
    (np.random.RandomState(42).rand(50, 200) > 0.5).astype(np.float64)
)


class MockTrainer(TrainerBase):
    def __init__(self) -> None:
        self.epoch: int = 0

    def load_state(self, ifs: IO) -> None:
        self.epoch = pickle.load(ifs)["epoch"]

    def save_state(self, ofs: IO) -> None:
        pickle.dump(dict(epoch=self.epoch), ofs)

    def run_epoch(self) -> None:
        self.epoch += 1


class MockRecommender(BaseRecommenderWithEarlyStopping, register_class=False):
    trainer_class = MockTrainer

    def __init__(
        self,
        X: InteractionMatrix,
        target_epoch: int = 20,
        target_score: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(X, **kwargs)
        self.target_epoch = target_epoch
        self.target_score = target_score
        self.rns = np.random.RandomState(42)

    def _create_trainer(self) -> MockTrainer:
        sleep(0.01)
        return MockTrainer()

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        score = np.zeros(*self.X_train_all.shape)
        return score

    def _current_score(self) -> float:
        assert isinstance(self.trainer, MockTrainer)
        coeff: float
        if self.trainer.epoch > self.target_epoch:
            coeff = 1 - (self.trainer.epoch - self.target_epoch) / self.target_epoch
        else:
            coeff = self.trainer.epoch / self.target_epoch
        return self.target_score * coeff


class MockEvaluator(Evaluator):
    def __init__(self, X: sps.csr_matrix) -> None:
        super().__init__(X, offset=0)
        self.target_metric = TargetMetric.ndcg
        self.cutoff = 30

    def get_score(self, model: BaseRecommender) -> Dict[str, float]:
        assert isinstance(model, MockRecommender)
        return {self.target_metric.name: model._current_score()}


class MockOptimizer(BaseOptimizerWithEarlyStopping):
    recommender_class = MockRecommender
    default_tune_range = [UniformSuggestion("target_score", 0, 1)]


@pytest.mark.parametrize("X, target_epoch", [(X_small, 20)])
def test_optimizer_by_mock(X: InteractionMatrix, target_epoch: int) -> None:

    evaluator = MockEvaluator(X)
    optimizer = MockOptimizer(
        X,
        evaluator,
        fixed_params=dict(target_epoch=target_epoch),
        logger=getLogger("IGNORE"),
    )
    with redirect_stdout(open(os.devnull, "w")):
        with redirect_stderr(open(os.devnull, "w")):
            config, history = optimizer.optimize(n_trials=20, random_seed=42)
    assert len(config) == 3
    assert config["max_epoch"] == target_epoch
    best_index = np.nanargmax(-history.value.values)
    best_target_score_inferred = history.target_score.iloc[best_index]
    best_ndcg = history["ndcg@30"].iloc[best_index]

    assert best_target_score_inferred == pytest.approx(best_ndcg)
    assert config["target_score"] == pytest.approx(best_target_score_inferred)
    assert np.all(best_target_score_inferred >= history.target_score.values)
