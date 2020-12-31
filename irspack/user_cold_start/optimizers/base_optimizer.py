import logging
from typing import Any, Dict, List, Optional, Type

import optuna
from sklearn.model_selection import train_test_split

from irspack.definitions import InteractionMatrix
from irspack.parameter_tuning import Suggestion, overwrite_suggestions
from irspack.user_cold_start.evaluator import UserColdStartEvaluator
from irspack.user_cold_start.recommenders.base import (
    BaseUserColdStartRecommender,
    ProfileMatrix,
)
from irspack.utils.default_logger import get_default_logger


class BaseOptimizer:
    default_tune_range: List[Suggestion]
    recommender_class: Type[BaseUserColdStartRecommender]

    def __init__(
        self,
        X_train: InteractionMatrix,
        profile_train: ProfileMatrix,
        evaluator: UserColdStartEvaluator,
        target_metric: str = "ndcg",
        suggest_overwrite: List[Suggestion] = list(),
        fixed_params: Dict[str, Any] = dict(),
        logger: Optional[logging.Logger] = None,
    ):
        if logger is None:
            logger = get_default_logger()

        self.logger = logger

        self.X_train = X_train
        self.profile_train = profile_train
        self.evaluator = evaluator
        self.target_metric = target_metric
        self.current_trial: int = 0
        self.best_trial_index: Optional[int] = None
        self.best_val = float("inf")
        self.best_params: Optional[Dict[str, Any]] = None
        self.learnt_config_best: Dict[str, Any] = dict()

        self.valid_results: List[Dict[str, float]] = []
        self.tried_configs: List[Dict[str, Any]] = []
        self.suggestions = overwrite_suggestions(
            self.default_tune_range, suggest_overwrite, fixed_params
        )
        self.fixed_params = fixed_params

    def _suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        parameters: Dict[str, Any] = dict()
        for s in self.suggestions:
            parameters[s.name] = s.suggest(trial)
        return parameters

    def optimize(
        self,
        n_trials: int = 20,
        timeout: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        def objective(trial: optuna.Trial) -> float:
            param_dict = dict(**self._suggest(trial), **self.fixed_params)
            recommender = self.recommender_class(
                self.X_train, self.profile_train, **param_dict
            )
            recommender.learn()
            val_score: float = self.evaluator.get_score(recommender)[self.target_metric]
            if (-val_score) < self.best_val:
                self.best_val = -val_score
                self.best_params = param_dict
                self.logger.info("Found best %s using this config.", self.target_metric)
                self.best_trial_index = self.current_trial

            return -val_score

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=random_seed)
        )
        study.optimize(objective, n_trials, timeout=timeout)
        if self.best_params is None:
            raise RuntimeError(
                "best parameter not found (possibly because no trial has been made)"
            )
        best_params: Dict[str, Any] = self.best_params
        return best_params

    @classmethod
    def split_and_optimize(
        cls,
        X_train: InteractionMatrix,
        X_profile: ProfileMatrix,
        n_trials: int = 20,
        target_metric: str = "ndcg",
        split_config: Dict[str, Any] = dict(test_size=0.2, random_state=42),
        evaluator_config: Dict[str, Any] = dict(),
        timeout: Optional[int] = None,
        suggest_overwrite: List[Suggestion] = [],
        fixed_params: Dict[str, Any] = dict(),
        logger: Optional[logging.Logger] = None,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        X_tt, X_tv, profile_tt, profile_tv = train_test_split(
            X_train, X_profile, **split_config
        )
        X_tt.sort_indices()
        X_tv.sort_indices()
        profile_tt.sort_indices()
        profile_tv.sort_indices()
        evaluator = UserColdStartEvaluator(X_tv, profile_tv, **evaluator_config)
        optimizer = cls(
            X_tt,
            profile_tt,
            evaluator,
            target_metric=target_metric,
            suggest_overwrite=suggest_overwrite,
            fixed_params=fixed_params,
            logger=logger,
        )
        return optimizer.optimize(
            n_trials=n_trials, timeout=timeout, random_seed=random_seed
        )
