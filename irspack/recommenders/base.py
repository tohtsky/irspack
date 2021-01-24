from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
from optuna.trial import Trial
from scipy import sparse as sps

if TYPE_CHECKING:
    from .. import evaluator

from ..definitions import (
    DenseMatrix,
    DenseScoreArray,
    InteractionMatrix,
    UserIndexArray,
)


def _sparse_to_array(U: Any) -> np.ndarray:
    if sps.issparse(U):
        return U.toarray()
    else:
        return U


class CallBeforeFitError(Exception):
    pass


class BaseRecommender(object, metaclass=ABCMeta):
    """The base class for all (hot) recommenders.

    Args:
        X_train_all (csr_matrix|csc_matrix|np.ndarray): user/item interaction matrix.
            each row correspods to a user's interaction with items.
    """

    def __init__(self, X_train_all: InteractionMatrix, **kwargs: Any) -> None:
        self.X_train_all = sps.csr_matrix(X_train_all).astype(np.float64)
        self.n_users: int = self.X_train_all.shape[0]
        self.n_items: int = self.X_train_all.shape[1]
        self.X_train_all.sort_indices()

        # this will store configurable parameters learnt during the training,
        # e.g., the epoch with the best validation score.
        self.learnt_config: Dict[str, Any] = dict()

    def learn(self) -> "BaseRecommender":
        """Learns and returns itself.

        Returns:
            The model after fitting process.
        """
        self._learn()
        return self

    @abstractmethod
    def _learn(self) -> None:
        raise NotImplementedError("_learn must be implemented.")

    def learn_with_optimizer(
        self, evaluator: Optional["evaluator.Evaluator"], trial: Optional[Trial]
    ) -> None:
        """Learning procedures with early stopping and pruning.

        Args:
            evaluator : The evaluator to measure the score.
            trial : The current optuna trial under the study (if any.)
        """
        # by default, evaluator & trial does not play any role.
        self.learn()

    @abstractmethod
    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        """Compute the item recommendation score for a subset of users.

        Args:
            user_indices : The index defines the subset of users.

        Returns:
            The item scores. Its shape will be (len(user_indices), self.n_items)
        """
        raise NotImplementedError("get_score must be implemented")  # pragma: no cover

    def get_score_block(self, begin: int, end: int) -> DenseScoreArray:
        """Compute the score for a block of the users.

        Args:
            begin (int): where the evaluated user block begins.
            end (int): where the evaluated user block ends.

        Returns:
            The item scores. Its shape will be (end - begin, self.n_items)
        """
        raise NotImplementedError("get_score_block not implemented!")

    def get_score_remove_seen(self, user_indices: UserIndexArray) -> DenseScoreArray:
        """Compute the item score and mask the item in the training set. Masked items will have the score -inf.

        Args:
            user_indices : Specifies the subset of users.

        Returns:
            The masked item scores. Its shape will be (len(user_indices), self.n_items)
        """
        scores = self.get_score(user_indices)
        if sps.issparse(scores):
            scores = scores.toarray()
        m = self.X_train_all[user_indices].tocsr()
        scores[m.nonzero()] = -np.inf
        if scores.dtype != np.float64:
            scores = scores.astype(np.float64)
        return scores

    def get_score_remove_seen_block(self, begin: int, end: int) -> DenseScoreArray:
        """Compute the score for a block of the users, and mask the items in the training set. Masked items will have the score -inf.

        Args:
            begin (int): where the evaluated user block begins.
            end (int): where the evaluated user block ends.

        Returns:
            The masked item scores. Its shape will be (end - begin, self.n_items)
        """
        scores = _sparse_to_array(self.get_score_block(begin, end))
        m = self.X_train_all[begin:end]
        scores[m.nonzero()] = -np.inf
        if scores.dtype != np.float64:
            scores = scores.astype(np.float64)
        return scores

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        """Compute the item recommendation score for unseen users whose profiles are given as another user-item relation matrix.

        Args:
            X : The profile user-item relation matrix for unseen users.
                Its number of rows is arbitrary, but the number of columns must be self.n_items.

        Returns:
            Computed item scores for users. Its shape is equal to X.
        """
        raise NotImplementedError(
            f"get_score_cold_user is not implemented for {self.__class__.__name__}!"
        )  # pragma: no cover

    def get_score_cold_user_remove_seen(self, X: InteractionMatrix) -> DenseScoreArray:
        """Compute the item recommendation score for unseen users whose profiles are given as another user-item relation matrix. The score will then be masked by the input.

        Args:
            X : The profile user-item relation matrix for unseen users.
                Its number of rows is arbitrary, but the number of columns must be self.n_items.

        Returns:
            Computed & masked item scores for users. Its shape is equal to X.
        """
        score = self.get_score_cold_user(X)
        score[X.nonzero()] = -np.inf
        return score


class BaseSimilarityRecommender(BaseRecommender):
    W_: Optional[Union[sps.csr_matrix, sps.csc_matrix, np.ndarray]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.W_ = None

    @property
    def W(self) -> Union[sps.csr_matrix, sps.csc_matrix, np.ndarray]:
        """Computed similarity weight matrix.

        Raises:
            RuntimeError: Raises When there is not W_ attributes (e.g., method call before the fit).

        Returns:
            The similarity matrix. Score will be computed by e.g. self.X.dot(self.W)
        """
        if self.W_ is None:
            raise RuntimeError("W fetched before fit.")
        return self.W_

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        return _sparse_to_array(self.X_train_all[user_indices].dot(self.W))

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        return _sparse_to_array(X.dot(self.W))

    def get_score_block(self, begin: int, end: int) -> DenseScoreArray:
        return _sparse_to_array(self.X_train_all[begin:end].dot(self.W))


class BaseUserSimilarityRecommender(BaseRecommender):
    U_: Optional[Union[sps.csr_matrix, sps.csc_matrix, np.ndarray]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._X_csc: sps.csc_matrix = self.X_train_all.tocsc()
        self.U_ = None

    @property
    def U(self) -> Union[sps.csr_matrix, sps.csc_matrix, np.ndarray]:
        """Computed similarity weight matrix.

        Raises:
            RuntimeError: Raises When there is not W_ attributes (e.g., method call before the fit).

        Returns:
            The similarity matrix. Score will be computed by e.g. self.X.dot(self.W)
        """
        if self.U_ is None:
            raise RuntimeError("W fetched before fit.")
        return self.U_

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        return _sparse_to_array(self.U[user_indices].dot(self._X_csc).toarray())

    def get_score_block(self, begin: int, end: int) -> DenseScoreArray:
        return _sparse_to_array(self.U[begin:end].dot(self._X_csc))


class BaseRecommenderWithUserEmbedding(BaseRecommender):
    """Defines a recommender with user embedding (e.g., matrix factorization.).
    These class can be a base CF estimator for CB2CF (with user profile -> user embedding NN).
    """

    @abstractmethod
    def get_user_embedding(
        self,
    ) -> DenseMatrix:
        """Get user embedding vectors.

        Returns:
            The latent vector representation of users.
            Its number of rows is equal to the number of the users.
        """

    @abstractmethod
    def get_score_from_user_embedding(
        self, user_embedding: DenseMatrix
    ) -> DenseScoreArray:
        """Compute the item score from user embedding. Mainly used for cold-start scenario.

        Args:
            user_embedding : Latent user representation obtained elsewhere.

        Returns:
            DenseScoreArray: The score array. Its shape will be ``(user_embedding.shape[0], self.n_items)``
        """
        raise NotImplementedError("get_score_from_item_embedding must be implemtented.")


class BaseRecommenderWithItemEmbedding(BaseRecommender):
    """Defines a recommender with item embedding (e.g., matrix factorization.).
    These class can be a base CF estimator for CB2CF (with item profile -> item embedding NN).
    """

    @abstractmethod
    def get_item_embedding(
        self,
    ) -> DenseMatrix:
        """Get item embedding vectors.

        Returns:
            The latent vector representation of items.
            Its number of rows is equal to the number of the items.
        """
        raise NotImplementedError(
            "get_item_embedding must be implemented."
        )  # pragma: no cover

    @abstractmethod
    def get_score_from_item_embedding(
        self, user_indices: UserIndexArray, item_embedding: DenseMatrix
    ) -> DenseScoreArray:
        raise NotImplementedError("get_score_from_item_embedding must be implemented.")
