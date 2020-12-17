from ..definitions import UserIndexArray
from typing import List, Dict, Any
from abc import ABC, abstractmethod

import optuna
import numpy as np
from sklearn.model_selection import train_test_split

from ..recommenders.base import DenseScoreArray, InteractionMatrix
from ..parameter_tuning import Suggestion
from ..user_cold_start.base import ProfileMatrix
from .evaluator import ItemColdStartEvaluator


class ItemColdStartRecommenderBase(ABC):
    suggest_param_range: List[Suggestion] = []

    def __init__(
        self, X_interaction: InteractionMatrix, X_profile: ProfileMatrix, **kwargs
    ):
        assert X_interaction.shape[1] == X_profile.shape[0]
        self.n_users = X_interaction.shape[0]
        self.n_items = X_interaction.shape[1]
        self.profile_dimension = X_profile.shape[1]
        self.X_profile = X_profile
        self.X_interaction = X_interaction.tocsr()
        self.X_interaction.sort_indices()
    
    def learn(self) -> "ItemColdStartRecommenderBase":
        self._learn()
        return self

    @abstractmethod
    def _learn(self) -> None:
        pass

    @abstractmethod
    def get_score_for_user_range(
        self, user_range: UserIndexArray, item_profile: ProfileMatrix
    ) -> DenseScoreArray:
        raise NotImplementedError("implemented in the descendant")

    @classmethod
    def optimize(
        cls,
        X_train: InteractionMatrix,
        X_profile: ProfileMatrix,
        n_trials: int,
        target_metric="ndcg",
        other_args=dict(),
        split_config: Dict[str, Any] = dict(test_size=0.2, random_state=42),
        timeout=None,
    ):
        train_index, val_index = train_test_split(
            np.arange(X_train.shape[1]), **split_config
        )
        X_tt = X_train[:, train_index].copy()
        X_tt.sort_indices()
        X_tv = X_train[:, val_index].copy()
        X_tv.sort_indices()
        profile_tt = X_profile[train_index, :]
        profile_tv = X_profile[val_index, :]
        profile_tt.sort_indices()
        profile_tv.sort_indices()
        evaluator_ = ItemColdStartEvaluator(X_tv, profile_tv)

        def objective(trial: optuna.Trial):
            param_dict = dict(**other_args)
            for suggestion in cls.suggest_param_range:
                param_dict[suggestion.name] = suggestion.suggest(trial)
            recommender = cls(X_tt, profile_tt, **param_dict)
            recommender.learn()
            result = evaluator_.get_score(recommender)[target_metric]
            return -result

        study = optuna.create_study()
        study.optimize(objective, n_trials, timeout=timeout)
        return study.best_params

