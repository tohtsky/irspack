from typing import Optional

from sklearn.decomposition import TruncatedSVD

from ..definitions import (
    DenseMatrix,
    DenseScoreArray,
    InteractionMatrix,
    UserIndexArray,
)
from ..parameter_tuning import IntegerSuggestion
from .base import (
    BaseRecommenderWithColdStartPredictability,
    BaseRecommenderWithUserEmbedding,
    BaseRecommenderWithItemEmbedding,
)


class TruncatedSVDRecommender(
    BaseRecommenderWithColdStartPredictability,
    BaseRecommenderWithUserEmbedding,
    BaseRecommenderWithItemEmbedding,
):
    default_tune_range = [IntegerSuggestion("n_components", 4, 512)]
    decomposer: Optional[TruncatedSVD]
    z: Optional[DenseMatrix]

    def __init__(self, X_all, n_components=4):
        super().__init__(X_all)
        self.n_components = n_components
        self.decomposer = None

    def learn(self):
        self.decomposer = TruncatedSVD(n_components=self.n_components)
        self.z = self.decomposer.fit_transform(self.X_all)

    def get_score(self, user_indices):
        return self.z[user_indices].dot(self.decomposer.components_)

    def get_score_block(self, begin, end):
        return self.z[begin:end].dot(self.decomposer.components_)

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        if self.decomposer is None:
            raise RuntimeError("No fit yet")
        return self.decomposer.transform(X).dot(self.decomposer.components_)

    def get_user_embedding(self) -> DenseMatrix:
        if self.z is None:
            raise RuntimeError("No fit yet")
        return self.z

    def get_score_from_user_embedding(
        self, user_embedding: DenseMatrix
    ) -> DenseScoreArray:
        if self.decomposer is None:
            raise RuntimeError("No fit yet")

        return user_embedding.dot(self.decomposer.components_)

    def get_item_embedding(self) -> DenseMatrix:
        if self.decomposer is None:
            raise RuntimeError("No fit yet")
        return self.decomposer.components_.T

    def get_score_from_item_embedding(
        self, user_indices: UserIndexArray, item_embedding: DenseMatrix
    ) -> DenseScoreArray:
        if self.z is None:
            raise RuntimeError("No fit yet")
        return self.z[user_indices].dot(item_embedding.T)
