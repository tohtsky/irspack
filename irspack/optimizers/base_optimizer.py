import logging
import numpy as np
import time
from abc import ABC
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type

import optuna
import pandas as pd

from ..parameter_tuning import overwrite_suggestions, Suggestion
from ..evaluator import Evaluator
from ..recommenders.base import (
    BaseRecommender,
    BaseRecommenderWithThreadingSupport,
    InteractionMatrix,
)
from ..recommenders.base_earlystop import BaseRecommenderWithEarlyStopping


class BaseOptimizer(ABC):
    recommender_class: Type[BaseRecommender]
    default_tune_range: List[Suggestion]

    def __init__(
        self,
        data: InteractionMatrix,
        val_evaluator: Evaluator,
        metric: str = "ndcg",
        logger: Optional[logging.Logger] = None,
        suggest_overwrite: List[Suggestion] = list(),
        fixed_params: Dict[str, Any] = dict(),
    ):
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
            logger.addHandler(handler)
        self.logger = logger
        self._data = data
        self.val_evaluator = val_evaluator
        self.metric = metric

        self.current_trial: int = 0
        self.best_trial_index: Optional[int] = None
        self.best_val = float("inf")
        self.best_params: Optional[Dict[str, Any]] = None
        self.learnt_config_best: Dict[str, Any] = dict()  # to store early-stopped epoch

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

    def get_model_arguments(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return args, kwargs

    def optimize(
        self, n_trials: int = 20, timeout: Optional[int] = None
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        self.logger.info(
            f"""Start parameter search for {self.recommender_class.__name__} over following range:"""
        )
        self.logger.info("\n".join([repr(s) for s in self.suggestions]))
        study = optuna.create_study()
        self.current_trial = -1
        self.best_val = float("inf")
        self.best_time = None
        self.valid_results = []
        self.tried_configs = []

        def objective_func(trial: optuna.Trial) -> float:
            self.current_trial += 1  # for pruning
            start = time.time()
            params = dict(**self._suggest(trial), **self.fixed_params)
            self.logger.info(f"\nTrial {self.current_trial}:")
            self.logger.info(f"parameter = {params}")

            arg, parameters = self.get_model_arguments(**params)

            self.tried_configs.append(parameters)
            recommender = self.recommender_class(self._data, *arg, **parameters)
            recommender.learn_with_optimizer(self.val_evaluator, trial)

            score = self.val_evaluator.get_score(recommender)
            end = time.time()

            time_spent = end - start
            score["time"] = time_spent
            self.valid_results.append(score)
            self.logger.info(
                f"Config {self.current_trial} obtained the following scores: {score} within {time_spent} seconds."
            )
            val_score = score[self.metric]
            if (-val_score) < self.best_val:
                self.best_val = -val_score
                self.best_time = time_spent
                self.best_params = parameters
                self.learnt_config_best = dict(**recommender.learnt_config)
                self.logger.info(f"Found best {self.metric} using this config.")
                self.best_trial_index = self.current_trial

            return -val_score

        study.optimize(objective_func, n_trials=n_trials, timeout=timeout)
        if self.best_params is None:
            raise RuntimeError("best parameter not found.")
        best_params = dict(**self.best_params)
        best_params.update(**self.learnt_config_best)
        self.best_params = best_params
        result_df = pd.concat(
            [
                pd.DataFrame(self.tried_configs),
                pd.DataFrame(self.valid_results),
            ],
            axis=1,
        ).copy()
        is_best = np.zeros(result_df.shape[0], dtype=np.bool)
        if self.best_trial_index is not None:
            is_best[self.best_trial_index] = True
        result_df["is_best"] = is_best
        return best_params, result_df


class BaseOptimizerWithEarlyStopping(BaseOptimizer):
    recommender_class: Type[BaseRecommenderWithEarlyStopping]

    def __init__(
        self,
        data: InteractionMatrix,
        val_evaluator: Evaluator,
        metric: str = "ndcg",
        logger: Optional[logging.Logger] = None,
        suggest_overwrite: List[Suggestion] = list(),
        fixed_params: Dict[str, Any] = dict(),
        max_epoch: int = 512,
        validate_epoch: int = 5,
        score_degradation_max: int = 5,
        **kwargs: Any,
    ):
        super().__init__(
            data,
            val_evaluator,
            metric,
            logger=logger,
            suggest_overwrite=suggest_overwrite,
            fixed_params=fixed_params,
        )
        self.max_epoch = max_epoch
        self.validate_epoch = validate_epoch
        self.score_degradation_max = score_degradation_max

    def get_model_arguments(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return super().get_model_arguments(
            *args,
            max_epoch=self.max_epoch,
            validate_epoch=self.validate_epoch,
            score_degradation_max=self.score_degradation_max,
            **kwargs,
        )


class BaseOptimizerWithThreadingSupport(BaseOptimizer):
    recommender_class: Type[BaseRecommenderWithThreadingSupport]

    def __init__(
        self,
        data: InteractionMatrix,
        val_evaluator: Evaluator,
        metric: str = "ndcg",
        logger: Optional[logging.Logger] = None,
        suggest_overwrite: List[Suggestion] = list(),
        fixed_params: Dict[str, Any] = dict(),
        n_thread: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(
            data,
            val_evaluator,
            metric,
            logger,
            suggest_overwrite=suggest_overwrite,
            fixed_params=fixed_params,
        )
        self.n_thread = n_thread

    def get_model_arguments(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return super().get_model_arguments(*args, n_thread=self.n_thread, **kwargs)
