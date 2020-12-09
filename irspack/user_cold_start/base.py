"""
Copyright 2020 BizReach, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import List, Union, Dict, Any
from abc import ABC, abstractmethod

import optuna
import numpy as np
from scipy import sparse as sps
from sklearn.model_selection import train_test_split

from ..recommenders.base import InteractionMatrix
from ..definitions import ProfileMatrix
from ..parameter_tuning import Suggestion
from .evaluator import UserColdStartEvaluator


class UserColdStartRecommenderBase(ABC):
    suggest_param_range: List[Suggestion] = []

    def __init__(
        self, X_interaction: InteractionMatrix, X_profile: ProfileMatrix, **kwargs
    ):
        assert X_interaction.shape[0] == X_profile.shape[0]
        self.n_users = X_profile.shape[0]
        self.n_items = X_interaction.shape[1]
        self.profile_dimension = X_profile.shape[1]
        self.X_profile = X_profile
        self.X_interaction = X_interaction
        pass

    @abstractmethod
    def learn(self) -> None:
        pass

    @abstractmethod
    def get_score(self, profile):
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
        X_tt, X_tv, profile_tt, profile_tv = train_test_split(
            X_train, X_profile, **split_config
        )
        X_tt.sort_indices()
        X_tv.sort_indices()
        profile_tt.sort_indices()
        profile_tv.sort_indices()
        evaluator_ = UserColdStartEvaluator(X_tv, profile_tv)

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
