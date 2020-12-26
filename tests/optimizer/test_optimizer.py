import numpy as np
import pytest
import scipy.sparse as sps

from irspack.dataset.movielens import MovieLens100KDataManager
from irspack.definitions import DenseScoreArray, UserIndexArray
from irspack.evaluator import Evaluator
from irspack.optimizers import BaseOptimizer
from irspack.parameter_tuning import (
    CategoricalSuggestion,
    IntegerLogUniformSuggestion,
    IntegerSuggestion,
    LogUniformSuggestion,
    UniformSuggestion,
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


class MockRecommender(BaseRecommender):
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


class MockOptimizer(BaseOptimizer):
    recommender_class = MockRecommender
    default_tune_range = [
        UniformSuggestion("p1", 0, 1),
        LogUniformSuggestion("reg", 0.99, 1.01),
        IntegerSuggestion("I1", 100, 102),
        IntegerLogUniformSuggestion("I2", 500, 502),
        CategoricalSuggestion("flag", ["foo", "bar"]),
    ]


@pytest.mark.parametrize("X", [X_small, X_large])
def test_optimizer_by_mock(X: sps.csr_matrix) -> None:
    X_train, X_val = rowwise_train_test_split(X, test_ratio=0.5, random_seed=0)
    evaluator = Evaluator(X_val, 0)
    optimizer = MockOptimizer(
        X_train, evaluator, logger=None, fixed_params=dict(X_test=X_val)
    )
    config, _ = optimizer.optimize(n_trials=40, random_seed=42)
    assert config["p1"] >= 0.9
    assert (config["reg"] >= 0.99) and (config["reg"] <= 1.01)
    assert (config["I1"] >= 100) and (config["I1"] <= 102)
    assert (config["I2"] >= 500) and (config["I2"] <= 502)
    assert config["flag"] in ["foo", "bar"]
