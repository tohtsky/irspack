from typing import Optional

from sklearn.decomposition import TruncatedSVD

from ..definitions import (
    DenseMatrix,
    DenseScoreArray,
    InteractionMatrix,
    UserIndexArray,
)
from .base import BaseRecommenderWithItemEmbedding, BaseRecommenderWithUserEmbedding


class TruncatedSVDRecommender(
    BaseRecommenderWithUserEmbedding,
    BaseRecommenderWithItemEmbedding,
):
    """Use (randomized) SVD to factorize the input matrix into low-rank matrices.

    Args:
        X_train_all (Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
            Input interaction matrix.

        n_components (int, optional):
            The rank of approximation. Defaults to 4.
            If this is larger than X_train_all, the value will be truncated into ``X_train_all.shape[1]``
    """

    def __init__(self, X_train_all: InteractionMatrix, n_components: int = 4) -> None:
        super().__init__(X_train_all)
        self.n_components = min(n_components, self.X_train_all.shape[1])
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
        self.z_ = self.decomposer_.fit_transform(self.X_train_all)

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
