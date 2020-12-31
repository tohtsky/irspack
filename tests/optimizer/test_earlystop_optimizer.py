import pickle
from logging import getLogger
from typing import IO, Any

import numpy as np
import pytest
import scipy.sparse as sps

from irspack.definitions import DenseScoreArray, InteractionMatrix, UserIndexArray
from irspack.evaluator import Evaluator
from irspack.optimizers.base_optimizer import BaseOptimizerWithEarlyStopping
from irspack.recommenders.base_earlystop import (
    BaseRecommenderWithEarlyStopping,
    TrainerBase,
)
from irspack.split import rowwise_train_test_split

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


class MockRecommender(BaseRecommenderWithEarlyStopping):
    trainer_class = MockTrainer

    def __init__(
        self,
        X: InteractionMatrix,
        X_test: InteractionMatrix,
        target_epoch: int = 20,
        **kwargs: Any
    ):
        super().__init__(X, **kwargs)
        self.answer = np.asfarray(X_test)
        self.target_epoch = target_epoch
        self.rns = np.random.RandomState(42)

    def _create_trainer(self) -> MockTrainer:
        return MockTrainer()

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        assert isinstance(self.trainer, MockTrainer)
        score = self.answer[user_indices]
        score = (
            score
            + 10
            * self.rns.randn(*score.shape)
            * abs(self.target_epoch - self.trainer.epoch)
            / self.target_epoch
        )
        return score


class MockOptimizer(BaseOptimizerWithEarlyStopping):
    recommender_class = MockRecommender


@pytest.mark.parametrize(
    "X, target_epoch", [(X_small, 20), (X_small, 15), (X_large, 5)]
)
def test_optimizer_by_mock(X: InteractionMatrix, target_epoch: int) -> None:
    X_train, X_val = rowwise_train_test_split(X)
    evaluator = Evaluator(X_val, 0)
    optimizer = MockOptimizer(
        X_train,
        evaluator,
        fixed_params=dict(X_test=X_val.toarray(), target_epoch=target_epoch),
        logger=getLogger("IGNORE"),
    )
    config, _ = optimizer.optimize(n_trials=1, random_seed=42)
    assert config["max_epoch"] == target_epoch
