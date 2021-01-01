import enum
from abc import abstractmethod
from typing import Optional, Union

import numpy as np

from ..definitions import InteractionMatrix
from ..utils import okapi_BM_25_weight, remove_diagonal, tf_idf_weight
from ._knn import (
    AsymmetricSimilarityComputer,
    CosineSimilarityComputer,
    JaccardSimilarityComputer,
    TverskyIndexComputer,
)
from .base import BaseRecommenderWithThreadingSupport, BaseSimilarityRecommender


class FeatureWeightingScheme(str, enum.Enum):
    NONE = "NONE"
    TF_IDF = "TF_IDF"
    BM_25 = "BM_25"


class BaseKNNRecommender(
    BaseSimilarityRecommender, BaseRecommenderWithThreadingSupport
):
    def __init__(
        self,
        X_train_all: InteractionMatrix,
        shrinkage: float = 0.0,
        top_k: int = 100,
        n_threads: Optional[int] = 1,
        feature_weighting: str = "NONE",
    ):
        super().__init__(X_train_all, n_threads=n_threads)
        self.shrinkage = shrinkage
        self.top_k = top_k
        self.feature_weighting = FeatureWeightingScheme(feature_weighting)

    @abstractmethod
    def _create_computer(
        self, X: InteractionMatrix
    ) -> Union[
        CosineSimilarityComputer,
        AsymmetricSimilarityComputer,
        JaccardSimilarityComputer,
        TverskyIndexComputer,
    ]:
        raise NotImplementedError("")

    def _learn(self) -> None:
        if self.feature_weighting == FeatureWeightingScheme.NONE:
            X_weighted = self.X_train_all
        elif self.feature_weighting == FeatureWeightingScheme.TF_IDF:
            X_weighted = tf_idf_weight(self.X_train_all)
        elif self.feature_weighting == FeatureWeightingScheme.BM_25:
            X_weighted = okapi_BM_25_weight(self.X_train_all)
        else:
            raise RuntimeError("Unknown weighting scheme.")

        computer = self._create_computer(X_weighted.T)
        self.W_ = remove_diagonal(
            computer.compute_similarity(self.X_train_all.T, self.top_k)
        ).tocsc()


class CosineKNNRecommender(BaseKNNRecommender):
    def __init__(
        self,
        X_train_all: InteractionMatrix,
        shrinkage: float = 0.0,
        normalize: bool = False,
        top_k: int = 100,
        feature_weighting: str = "NONE",
        n_threads: Optional[int] = 1,
    ) -> None:
        super().__init__(
            X_train_all,
            shrinkage,
            top_k,
            n_threads,
            feature_weighting=feature_weighting,
        )
        self.normalize = normalize

    def _create_computer(self, X: InteractionMatrix) -> CosineSimilarityComputer:
        return CosineSimilarityComputer(
            X, self.shrinkage, self.normalize, self.n_threads
        )


class TverskyIndexKNNRecommender(BaseKNNRecommender):
    def __init__(
        self,
        X_train_all: InteractionMatrix,
        shrinkage: float = 0.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        top_k: int = 100,
        feature_weighting: str = "NONE",
        n_threads: Optional[int] = 1,
    ) -> None:
        super().__init__(
            X_train_all,
            shrinkage,
            top_k,
            n_threads,
            feature_weighting=feature_weighting,
        )
        self.alpha = alpha
        self.beta = beta

    def _create_computer(self, X: InteractionMatrix) -> TverskyIndexComputer:
        return TverskyIndexComputer(
            X, self.shrinkage, self.alpha, self.beta, self.n_threads
        )


class JaccardKNNRecommender(BaseKNNRecommender):
    def _create_computer(self, X: InteractionMatrix) -> JaccardSimilarityComputer:
        return JaccardSimilarityComputer(X, self.shrinkage, self.n_threads)


class AsymmetricCosineKNNRecommender(BaseKNNRecommender):
    def __init__(
        self,
        X_train_all: InteractionMatrix,
        shrinkage: float = 0.0,
        alpha: float = 0.5,
        top_k: int = 100,
        feature_weighting: str = "NONE",
        n_threads: Optional[int] = 1,
    ):
        super().__init__(
            X_train_all,
            shrinkage,
            top_k,
            n_threads,
            feature_weighting=feature_weighting,
        )
        self.alpha = alpha

    def _create_computer(self, X: InteractionMatrix) -> AsymmetricSimilarityComputer:
        return AsymmetricSimilarityComputer(
            X, self.shrinkage, self.alpha, self.n_threads
        )
