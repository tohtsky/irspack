from typing import Optional, Union
from .base import BaseRecommenderWithThreadingSupport, BaseSimilarityRecommender
from ..definitions import InteractionMatrix
from abc import abstractmethod
from ._knn import (
    CosineKNNComputer,
    AsymmetricSimilarityComputer,
    JaccardSimilarityComputer,
)


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
        CosineKNNComputer,
        AsymmetricSimilarityComputer,
        JaccardSimilarityComputer,
    ]:
        raise NotImplementedError("")

    def learn(self):
        computer = self._create_computer()
        self.W = computer.compute_block(self.X_all, self.top_k)
