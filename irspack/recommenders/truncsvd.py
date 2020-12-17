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

    def __init__(self, X_all: InteractionMatrix, n_components: int = 4) -> None:
        super().__init__(X_all)
        self.n_components = n_components
        self.decomposer_: Optional[TruncatedSVD] = None
        self.z_: Optional[DenseMatrix] = None

    @property
    def z(self) -> DenseMatrix:
        if self.z_ is None:
            raise RuntimeError("z fetched before fit")
        return self.z_

    @property
    def decomposer(self) -> TruncatedSVD:
        if self.decomposer_ is None:
            raise RuntimeError("decomposer fetched before fit.")
        return self.decomposer_

    def _learn(self) -> None:
        self.decomposer_ = TruncatedSVD(n_components=self.n_components)
        self.z_ = self.decomposer_.fit_transform(self.X_all)

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        return self.z[user_indices].dot(self.decomposer.components_)

    def get_score_block(self, begin: int, end: int) -> DenseScoreArray:
        return self.z[begin:end].dot(self.decomposer.components_)

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        return self.decomposer.transform(X).dot(self.decomposer.components_)

    def get_user_embedding(self) -> DenseMatrix:
        return self.z

    def get_score_from_user_embedding(
        self, user_embedding: DenseMatrix
    ) -> DenseScoreArray:
        if self.decomposer is None:
            raise RuntimeError("No fit yet")

        return user_embedding.dot(self.decomposer.components_)

    def get_item_embedding(self) -> DenseMatrix:
        return self.decomposer.components_.T

    def get_score_from_item_embedding(
        self, user_indices: UserIndexArray, item_embedding: DenseMatrix
    ) -> DenseScoreArray:
        return self.z[user_indices].dot(item_embedding.T)
