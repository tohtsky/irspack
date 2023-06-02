import logging
from abc import ABCMeta, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    no_type_check,
)

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import sparse as sps

if TYPE_CHECKING:
    from optuna import Study, Trial

    from .. import evaluation


ParameterSuggestFunctionType = Callable[["Trial"], Dict[str, Any]]

from ..definitions import (
    DenseMatrix,
    DenseScoreArray,
    InteractionMatrix,
    UserIndexArray,
)
from ..optimization.parameter_range import ParameterRange

R = TypeVar("R", bound="BaseRecommender")


def _sparse_to_array(U: Any) -> np.ndarray:
    res: np.ndarray
    if sps.issparse(U):
        res = U.toarray()
        return res
    else:
        res = U
        return res


class CallBeforeFitError(Exception):
    pass


class RecommenderConfig(BaseModel):
    class Config:
        extra = "forbid"


class RecommenderMeta(ABCMeta):
    recommender_name_vs_recommender_class: Dict[str, "RecommenderMeta"] = {}

    @no_type_check
    def __new__(
        mcs,
        name,
        bases,
        namespace,
        register_class: bool = True,
        **kwargs,
    ):

        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        if register_class:
            mcs.recommender_name_vs_recommender_class[name] = cls
        return cls


class BaseRecommender(object, metaclass=RecommenderMeta):
    """The base class for all (hot) recommenders.

    Args:
        X_train_all (csr_matrix|csc_matrix|np.ndarray): user/item interaction matrix.
            each row correspods to a user's interaction with items.
    """

    config_class: Type[RecommenderConfig]
    default_tune_range: List[ParameterRange]

    X_train_all: sps.csr_matrix
    """The matrix to feed into recommender."""

    def __init__(self, X_train_all: InteractionMatrix, **kwargs: Any) -> None:
        self.X_train_all: sps.csr_matrix = sps.csr_matrix(X_train_all).astype(
            np.float64
        )

        self.n_users: int = self.X_train_all.shape[0]
        self.n_items: int = self.X_train_all.shape[1]
        self.X_train_all.sort_indices()

        # this will store configurable parameters learnt during the training,
        # e.g., the epoch with the best validation score.
        self.learnt_config: Dict[str, Any] = dict()

    @classmethod
    def from_config(
        cls: Type[R],
        X_train_all: InteractionMatrix,
        config: RecommenderConfig,
    ) -> R:
        if not isinstance(config, cls.config_class):
            raise ValueError(
                f"Different config has been given. config must be {cls.config_class}"
            )
        return cls(X_train_all, **config.dict())

    def learn(self: R) -> R:
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
        self,
        evaluator: Optional["evaluation.Evaluator"],
        trial: Optional["Trial"],
        max_epoch: int = 128,
        validate_epoch: int = 5,
        score_degradation_max: int = 5,
    ) -> None:
        r"""Learning procedures with early stopping and pruning.

        Args:
            evaluator : The evaluator to measure the score.
            trial : The current optuna trial under the study (if any.)
            max_epoch:
                Maximal number of epochs.
                If iterative learning procedure is not available, this parameter will be ignored.
                Defaults to 128.
            validate_epoch:
                The frequency of validation score measurement.
                If iterative learning procedure is not available, this parameter will be ignored.
                Defaults to 5.
            validate_epoch:
                The frequency of validation score measurement.
                If iterative learning procedure is not available, this parameter will be ignored.
                Defaults to 5.
            score_degradation_max:
                Maximal number of allowed score degradation.
                If iterative learning procedure is not available, this parameter will be ignored.
                Defaults to 5.
        """
        # by default, evaluator & trial does not play any role.
        self.learn()

    @classmethod
    def default_suggest_parameter(
        cls, trial: "Trial", fixed_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            s.name: s.suggest(trial)
            for s in cls.default_tune_range
            if s.name not in fixed_params.keys()
        }

    @classmethod
    def tune(
        cls,
        data: Union[InteractionMatrix, None],
        evaluator: "evaluation.Evaluator",
        n_trials: int = 20,
        timeout: Optional[int] = None,
        data_suggest_function: Optional[Callable[["Trial"], InteractionMatrix]] = None,
        parameter_suggest_function: Optional[ParameterSuggestFunctionType] = None,
        fixed_params: Dict[str, Any] = dict(),
        random_seed: Optional[int] = None,
        prunning_n_startup_trials: int = 10,
        max_epoch: int = 128,
        validate_epoch: int = 5,
        score_degradation_max: int = 5,
        logger: Optional[logging.Logger] = None,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        r"""Perform the optimization step.
        `optuna.Study` object is created inside this function.

        Args:
            data:
                The training data.
                You can also provide tunable parameter dependent training data by providing `data_suggest_function`.
                In that case, data must be `None`.
            evaluator:
                The validation evaluator that measures the performance of the recommenders.
            n_trials:
                The number of expected trials (including pruned ones). Defaults to 20.
            timeout:
                If set to some value (in seconds), the study will exit after that time period.
                Note that the running trials is not interrupted, though. Defaults to `None`.
            data_suggest_function:
                If not `None`, this must be a function which takes `optuna.Trial` as its argument and returns training data. Defaults to `None`.
            parameter_suggest_function:
                If not `None`, this must be a function which takes `optuna.Trial` as its argument and returns `Dict[str, Any]` (i.e., some keyword arguments of the recommender class).
                If `None`, `cls.default_suggest_parameter` will be used for the parameter suggestion.
                Defaults to `None`.
            fixed_params:
                Fixed parameters passed to recommenders during the optimization procedure.
                This will replace the suggested parameter (either by `cls.default_suggest_parameter` or `parameter_suggest_function`).
                Defaults to `dict()`.
            random_seed:
                The random seed to control `optuna.samplers.TPESampler`. Defaults to `None`.
            prunning_n_startup_trials:
                `n_startup_trials` argument passed to the constructor of `optuna.pruners.MedianPruner`.
            max_epoch:
                The maximal number of epochs for the training.
                If iterative learning procedure is not available, this parameter will be ignored.
            validate_epoch (int, optional):
                The frequency of validation score measurement.
                If iterative learning procedure is not available, this parameter will be ignored.
                Defaults to 5.
            score_degradation_max (int, optional):
                Maximal number of allowed score degradation.
                If iterative learning procedure is not available, this parameter will be ignored.
                Defaults to 5. Defaults to 5.
        Returns:
            A tuple that consists of

                1. A dict containing the best paramaters.
                   This dict can be passed to the recommender as ``**kwargs``.
                2. A ``pandas.DataFrame`` that contains the history of optimization.
        """

        from optuna import create_study, pruners, samplers

        study = create_study(
            sampler=samplers.TPESampler(seed=random_seed),
            pruner=pruners.MedianPruner(n_startup_trials=prunning_n_startup_trials),
        )
        return cls.tune_with_study(
            study,
            data=data,
            evaluator=evaluator,
            n_trials=n_trials,
            timeout=timeout,
            data_suggest_function=data_suggest_function,
            parameter_suggest_function=parameter_suggest_function,
            fixed_params=fixed_params,
            max_epoch=max_epoch,
            validate_epoch=validate_epoch,
            score_degradation_max=score_degradation_max,
            logger=logger,
        )

    @classmethod
    def tune_with_study(
        cls,
        study: "Study",
        data: Union[InteractionMatrix, None],
        evaluator: "evaluation.Evaluator",
        n_trials: int = 20,
        timeout: Optional[int] = None,
        data_suggest_function: Optional[Callable[["Trial"], InteractionMatrix]] = None,
        parameter_suggest_function: Optional[ParameterSuggestFunctionType] = None,
        fixed_params: Dict[str, Any] = dict(),
        max_epoch: int = 128,
        validate_epoch: int = 5,
        score_degradation_max: int = 5,
        logger: Optional[logging.Logger] = None,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        from ..optimization.optimizer import Optimizer

        if data is None:
            if data_suggest_function is None:
                raise ValueError(
                    "Either `data` or `data_sugget_function` must be provided."
                )
            _data_suggest_function = data_suggest_function
        else:

            def _data_suggest_function(trial: "Trial") -> InteractionMatrix:
                return data

        if parameter_suggest_function is not None:
            _parameter_suggest_function = parameter_suggest_function
        else:

            def _parameter_suggest_function(trial: "Trial") -> Dict[str, Any]:
                return cls.default_suggest_parameter(trial, fixed_params)

        optim = Optimizer(
            _data_suggest_function,
            _parameter_suggest_function,
            fixed_params,
            evaluator,
            max_epoch=max_epoch,
            validate_epoch=validate_epoch,
            score_degradation_max=score_degradation_max,
            logger=logger,
        )
        return optim.optimize_with_study(study, cls, n_trials=n_trials, timeout=timeout)

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
            scores = sps.csr_matrix(scores).toarray()
        m = self.X_train_all[user_indices].tocsr()
        scores[m.nonzero()] = -np.inf
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
    """The computed item-item similarity. Might not be initialized before `learn()` is called."""

    _W: Optional[Union[sps.csr_matrix, sps.csc_matrix, np.ndarray]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._W = None

    @property
    def W(self) -> Union[sps.csr_matrix, sps.csc_matrix, np.ndarray]:
        """The computed item-item similarity weight matrix."""
        if self._W is None:
            raise RuntimeError("W fetched before fit.")
        return self._W

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        return _sparse_to_array(self.X_train_all[user_indices].dot(self.W))

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        return _sparse_to_array(X.dot(self.W))

    def get_score_block(self, begin: int, end: int) -> DenseScoreArray:
        return _sparse_to_array(self.X_train_all[begin:end].dot(self.W))


class BaseUserSimilarityRecommender(BaseRecommender):
    """The computed user-user similarity. Might not be initialized before `learn()` is called."""

    U_: Optional[Union[sps.csr_matrix, sps.csc_matrix, np.ndarray]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._X_csc: sps.csc_matrix = self.X_train_all.tocsc()
        self.U_ = None

    @property
    def U(self) -> Union[sps.csr_matrix, sps.csc_matrix, np.ndarray]:
        """The computed user-user similarity weight matrix."""
        if self.U_ is None:
            raise RuntimeError("W fetched before fit.")
        return self.U_

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        return _sparse_to_array(self.U[user_indices].dot(self._X_csc).toarray())

    def get_score_block(self, begin: int, end: int) -> DenseScoreArray:
        return _sparse_to_array(self.U[begin:end].dot(self._X_csc))


class BaseRecommenderWithUserEmbedding:
    r"""Defines a recommender with user embedding (e.g., matrix factorization.)."""

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


class BaseRecommenderWithItemEmbedding:
    """Defines a recommender with item embedding (e.g., matrix factorization.)."""

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


def get_recommender_class(recommender_name: str) -> Type[BaseRecommender]:

    r"""Get recommender class from its class name.

    Args:
        recommender_name: The class name of the recommender.

    Returns:
        The recommender class with its class name being `recommender_name`.
    """
    result: Type[
        BaseRecommender
    ] = RecommenderMeta.recommender_name_vs_recommender_class[recommender_name]
    return result
