import warnings
from typing import Optional

from numpy import random
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
        X_train_all:
            Input interaction matrix.

        n_components:
            The rank of approximation. Defaults to 4.
            If this is larger than X_train_all, the value will be truncated into ``X_train_all.shape[1]``

        random_seed:
            The random seed to be passed on core TruncSVD.
    """

    def __init__(
        self,
        X_train_all: InteractionMatrix,
        n_components: int = 4,
        random_seed: int = 0,
    ) -> None:

        assert X_train_all.shape[1] > 1
        super().__init__(X_train_all)
        if n_components >= self.X_train_all.shape[1]:
            warnings.warn(
                "n_components >= than X_train_all.shape[1]. Set it to X_train_all.shape[1] - 1."
            )
            n_components = self.X_train_all.shape[1] - 1
        self.n_components = n_components
        self.decomposer_: Optional[TruncatedSVD] = None
        self.z_: Optional[DenseMatrix] = None
        self.random_seed = random_seed

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
        self.decomposer_ = TruncatedSVD(
            n_components=self.n_components, random_state=self.random_seed
        )
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
        return user_embedding.dot(self.decomposer.components_)

    def get_item_embedding(self) -> DenseMatrix:
        return self.decomposer.components_.T

    def get_score_from_item_embedding(
        self, user_indices: UserIndexArray, item_embedding: DenseMatrix
    ) -> DenseScoreArray:
        return self.z[user_indices].dot(item_embedding.T)
