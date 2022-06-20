from typing import List

import numpy as np
import pytest
import scipy.sparse as sps

from irspack.dataset.movielens import MovieLens100KDataManager
from irspack.definitions import DenseScoreArray, UserIndexArray
from irspack.evaluation import Evaluator
from irspack.optimization.parameter_range import (
    CategoricalRange,
    LogUniformFloatRange,
    LogUniformIntegerRange,
    ParameterRange,
    UniformFloatRange,
    UniformIntegerRange,
)
from irspack.recommenders import BaseRecommender
from irspack.split import rowwise_train_test_split

X_small = sps.csr_matrix(
    (np.random.RandomState(42).rand(100, 32) > 0.8).astype(np.float64)
)
ml_100k_df = MovieLens100KDataManager(force_download=True).read_interaction()
_, user_index = np.unique(ml_100k_df.userId, return_inverse=True)
_, movie_index = np.unique(ml_100k_df.movieId, return_inverse=True)
X_large = sps.csr_matrix(
    (np.ones(ml_100k_df.shape[0], dtype=np.float64), (user_index, movie_index)),
)


class MockRecommender(BaseRecommender, register_class=False):
    default_tune_range: List[ParameterRange] = [
        UniformFloatRange("p1", 0, 1),
        LogUniformFloatRange("reg", 0.99, 1.01),
        UniformIntegerRange("I1", 100, 102),
        LogUniformIntegerRange("I2", 500, 502),
        CategoricalRange("flag", ["foo", "bar"]),
    ]

    def __init__(
        self,
        X: sps.csr_matrix,
        X_test: sps.csr_matrix,
        p1: float = 1,
        I1: int = 1,
        I2: int = 1,
        reg: float = 1.0,
        flag: str = "hoge",
    ):
        super().__init__(X)
        self.p1 = p1  # only p1 matters
        self.I1 = I1
        self.I2 = I2
        self.reg = reg
        self.answer = X_test
        self.flag = flag
        self.rns = np.random.RandomState(0)

    def _learn(self) -> None:
        pass

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        score = self.answer[user_indices] * self.p1
        score = score + 10 * self.rns.rand(*score.shape) * (1 - self.p1)
        return score


@pytest.mark.parametrize("X", [X_small, X_large])
def test_optimizer_by_mock(X: sps.csr_matrix) -> None:
    X_train, X_val = rowwise_train_test_split(X, test_ratio=0.5, random_state=0)
    evaluator = Evaluator(X_val, 0)

    config, _ = MockRecommender.tune(
        X_train, evaluator, n_trials=40, random_seed=42, fixed_params=dict(X_test=X_val)
    )
    assert config["p1"] >= 0.9
    assert (config["reg"] >= 0.99) and (config["reg"] <= 1.01)
    assert (config["I1"] >= 100) and (config["I1"] <= 102)
    assert (config["I2"] >= 500) and (config["I2"] <= 502)
    assert config["flag"] in ["foo", "bar"]
