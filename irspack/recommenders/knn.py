import enum
from typing import Optional, Union
from .base import BaseRecommenderWithThreadingSupport, BaseSimilarityRecommender
from ..definitions import InteractionMatrix
import numpy as np
from abc import abstractmethod
from ._knn import (
    CosineSimilarityComputer,
    AsymmetricSimilarityComputer,
    JaccardSimilarityComputer,
    TverskyIndexComputer,
)
from ..parameter_tuning import (
    IntegerSuggestion,
    LogUniformSuggestion,
    UniformSuggestion,
    CategoricalSuggestion,
)
from ..utils import tf_idf_weight, okapi_BM_25_weight


class FeatureWeightingScheme(str, enum.Enum):
    NONE = "NONE"
    TF_IDF = "TF_IDF"
    BM_25 = "BM_25"


default_tune_range_knn = [
    IntegerSuggestion("top_k", 4, 1000),
    UniformSuggestion("shrinkage", 0, 1000),
    CategoricalSuggestion("feature_weighting", ["NONE", "TF_IDF", "BM_25"]),
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
        feature_weighting: str = "NONE",
    ):
        super().__init__(X_all, n_thread=n_thread)
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
    ]:
        raise NotImplementedError("")

    def learn(self):
        if self.feature_weighting == FeatureWeightingScheme.NONE:
            X_weighted = self.X_all
        elif self.feature_weighting == FeatureWeightingScheme.TF_IDF:
            X_weighted = tf_idf_weight(self.X_all)
        elif self.feature_weighting == FeatureWeightingScheme.BM_25:
            X_weighted = okapi_BM_25_weight(self.X_all)
        else:
            raise RuntimeError("Unknown weighting scheme.")

        computer = self._create_computer(X_weighted.T)
        self.W = computer.compute_similarity(self.X_all.T, self.top_k).tocsc()
        self.W[np.arange(self.n_item), np.arange(self.n_item)] = 0.0


class CosineKNNRecommender(BaseKNNRecommender):
    default_tune_range = default_tune_range_knn.copy() + [
        CategoricalSuggestion("normalize", [False, True])
    ]

    def __init__(
        self,
        X_all: InteractionMatrix,
        shrinkage: float = 0.0,
        normalize: bool = False,
        top_k: int = 100,
        feature_weighting: str = "NONE",
        n_thread: Optional[int] = 1,
    ):
        super().__init__(
            X_all,
            shrinkage,
            top_k,
            n_thread,
            feature_weighting=feature_weighting,
        )
        self.normalize = normalize

    def _create_computer(self, X) -> CosineSimilarityComputer:
        return CosineSimilarityComputer(
            X, self.shrinkage, self.normalize, self.n_thread
        )


class TverskyIndexKNNRecommender(BaseKNNRecommender):
    default_tune_range = default_tune_range_knn.copy() + [
        UniformSuggestion("alpha", 0, 2),
        UniformSuggestion("beta", 0, 2),
    ]

    def __init__(
        self,
        X_all: InteractionMatrix,
        shrinkage: float = 0.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        top_k: int = 100,
        feature_weighting: str = "NONE",
        n_thread: Optional[int] = 1,
    ):
        super().__init__(
            X_all,
            shrinkage,
            top_k,
            n_thread,
            feature_weighting=feature_weighting,
        )
        self.alpha = alpha
        self.beta = beta

    def _create_computer(self, X) -> CosineSimilarityComputer:
        return TverskyIndexComputer(
            X, self.shrinkage, self.alpha, self.beta, self.n_thread
        )


class JaccardKNNRecommender(BaseKNNRecommender):
    default_tune_range = default_tune_range_knn.copy()

    def _create_computer(
        self, X: InteractionMatrix
    ) -> JaccardSimilarityComputer:
        return JaccardSimilarityComputer(X, self.shrinkage, self.n_thread)


class AsymmetricCosineKNNRecommender(BaseKNNRecommender):
    default_tune_range = default_tune_range_knn + [
        UniformSuggestion("alpha", 0, 1)
    ]

    def __init__(
        self,
        X_all: InteractionMatrix,
        shrinkage: float = 0.0,
        alpha: float = 0.5,
        top_k: int = 100,
        feature_weighting: str = "NONE",
        n_thread: Optional[int] = 1,
    ):
        super().__init__(
            X_all,
            shrinkage,
            top_k,
            n_thread,
            feature_weighting=feature_weighting,
        )
        self.alpha = alpha

    def _create_computer(
        self, X: InteractionMatrix
    ) -> AsymmetricSimilarityComputer:
        return AsymmetricSimilarityComputer(
            X, self.shrinkage, self.alpha, self.n_thread
        )
