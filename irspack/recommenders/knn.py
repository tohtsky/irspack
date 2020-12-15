from typing import Optional, Union
from .base import BaseRecommenderWithThreadingSupport, BaseSimilarityRecommender
from ..definitions import InteractionMatrix
from abc import abstractmethod
from ._knn import (
    CosineSimilarityComputer,
    AsymmetricSimilarityComputer,
    JaccardSimilarityComputer,
)
from ..parameter_tuning import (
    IntegerSuggestion,
    LogUniformSuggestion,
    UniformSuggestion,
)

default_tune_range_knn = [
    IntegerSuggestion("top_k", 4, 1000),
    LogUniformSuggestion("shrinkage", 1e-10, 1e5),
]


class BaseKNNRecommender(
    BaseSimilarityRecommender, BaseRecommenderWithThreadingSupport
):
    def __init__(
        self,
        X_all: InteractionMatrix,
        shrinkage: float = 0.0,
        top_k: int = 100,
        n_thread: Optional[int] = 1,
    ):
        super().__init__(X_all, n_thread=n_thread)
        self.shrinkage = shrinkage
        self.top_k = top_k

    @abstractmethod
    def _create_computer(
        self,
    ) -> Union[
        CosineSimilarityComputer,
        AsymmetricSimilarityComputer,
        JaccardSimilarityComputer,
    ]:
        raise NotImplementedError("")

    def learn(self):
        computer = self._create_computer()
        self.W = computer.compute_similarity(self.X_all.T, self.top_k).tocsc()


class CosineKNNRecommender(BaseKNNRecommender):
    default_tune_range = default_tune_range_knn.copy()

    def _create_computer(
        self,
    ) -> CosineSimilarityComputer:
        return CosineSimilarityComputer(self.X_all.T, self.shrinkage, self.n_thread)


class JaccardKNNRecommender(BaseKNNRecommender):
    default_tune_range = default_tune_range_knn.copy()

    def _create_computer(
        self,
    ) -> JaccardSimilarityComputer:
        return JaccardSimilarityComputer(self.X_all.T, self.shrinkage, self.n_thread)


class AsymmetricCosineKNNRecommender(BaseKNNRecommender):
    default_tune_range = default_tune_range_knn + [UniformSuggestion("alpha", 0, 1)]

    def __init__(
        self,
        X_all: InteractionMatrix,
        shrinkage: float = 0.0,
        alpha: float = 0.5,
        top_k: int = 100,
        n_thread: Optional[int] = 1,
    ):
        super().__init__(X_all, shrinkage, top_k, n_thread)
        self.alpha = alpha

    def _create_computer(
        self,
    ) -> AsymmetricSimilarityComputer:
        return AsymmetricSimilarityComputer(
            self.X_all.T, self.shrinkage, self.alpha, self.n_thread
        )
