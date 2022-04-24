import pickle
import time
from typing import IO, Any, Dict, Sequence

import numpy as np
import scipy.sparse as sps
from scipy.special import expit

from irspack import (
    BaseOptimizer,
    BaseRecommender,
    DenseScoreArray,
    InteractionMatrix,
    TargetMetric,
    UserIndexArray,
)
from irspack.evaluator.evaluator import Evaluator
from irspack.optimizers.base_optimizer import BaseOptimizerWithEarlyStopping
from irspack.parameter_tuning import IntegerSuggestion, LogUniformSuggestion, Suggestion
from irspack.recommenders.base_earlystop import (
    BaseRecommenderWithEarlyStopping,
    TrainerBase,
)


class AutopilotMockRecommender(BaseRecommender, register_class=False):
    def __init__(self, X: InteractionMatrix, wait_time: float = 1.0):
        super().__init__(X)
        self.wait_time = wait_time

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        raise NotImplementedError("This is a mock")

    def _learn(self) -> None:
        time.sleep(self.wait_time)


class AutopilotMockOptimizer(BaseOptimizer):
    recommender_class = AutopilotMockRecommender
    default_tune_range: Sequence[Suggestion] = [
        LogUniformSuggestion("wait_time", 1e-2, 4.0)
    ]

    @classmethod
    def tune_range_given_memory_budget(
        cls, X: InteractionMatrix, memory_in_mb: int
    ) -> Sequence[Suggestion]:
        return []


class AutopilotMockEarlyStoppableTrainer(TrainerBase):
    def __init__(self, wait_time: float) -> None:
        self.epoch: int = 0
        self.wait_time = wait_time

    def load_state(self, ifs: IO) -> None:
        self.epoch = pickle.load(ifs)["epoch"]

    def save_state(self, ofs: IO) -> None:
        pickle.dump(dict(epoch=self.epoch), ofs)

    def run_epoch(self) -> None:
        time.sleep(self.wait_time)
        self.epoch += 1


class AutopilotMockEarlyStoppableRecommender(
    BaseRecommenderWithEarlyStopping, register_class=False
):
    trainer_class = AutopilotMockEarlyStoppableTrainer

    def __init__(
        self,
        X_train_all: InteractionMatrix,
        wait_time: float,
        target_epoch: int,
        **kwargs: Any
    ):
        self.wait_time = wait_time
        self.target_epoch = target_epoch
        super().__init__(X_train_all)

    def _create_trainer(self) -> TrainerBase:
        return AutopilotMockEarlyStoppableTrainer(self.wait_time)

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        raise NotImplementedError("This is a mock")


class AutopilotMockEarlyStoppableOptimizer(BaseOptimizerWithEarlyStopping):
    recommender_class = AutopilotMockEarlyStoppableRecommender
    default_tune_range = [
        IntegerSuggestion("target_epoch", 5, 30, 5),
        LogUniformSuggestion("wait_time", 1e-2, 1e2),
    ]


class AutopilotMockEvaluator(Evaluator):
    def __init__(self, X: sps.csr_matrix) -> None:
        super().__init__(X)
        self.target_metric = TargetMetric.ndcg
        self.cutoff = 10

    def get_score(self, model: BaseRecommender) -> Dict[str, float]:
        if isinstance(model, AutopilotMockRecommender):
            return {self.target_metric.name: expit(model.wait_time)}
        elif isinstance(model, AutopilotMockEarlyStoppableRecommender):
            assert model.trainer is not None
            assert isinstance(model.trainer, AutopilotMockEarlyStoppableTrainer)
            score = np.exp(-abs(model.trainer.epoch - model.target_epoch))
            return {self.target_metric.name: score}

        assert False, "should not happern"
