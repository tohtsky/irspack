import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid1

import optuna
import pandas as pd

from irspack.definitions import InteractionMatrix
from irspack.evaluator import Evaluator
from irspack.parameter_tuning import IntegerSuggestion, LogUniformSuggestion, Suggestion
from irspack.recommenders.ials import IALSRecommender

from ._optimizers import _get_maximal_n_components_for_budget
from .base_optimizer import BaseOptimizerWithEarlyStopping


class IALSOptimizer(BaseOptimizerWithEarlyStopping):
    default_tune_range = [
        IntegerSuggestion("n_components", 4, 300),
        LogUniformSuggestion("alpha0", 3e-3, 1),
        LogUniformSuggestion("reg", 1e-4, 1e-1),
    ]
    recommender_class = IALSRecommender

    def __init__(
        self,
        data: InteractionMatrix,
        val_evaluator: Evaluator,
        logger: Optional[logging.Logger] = None,
        suggest_overwrite: Sequence[Suggestion] = [],
        fixed_params: Dict[str, Any] = {},
        max_epoch: int = 16,
        validate_epoch: int = 1,
        score_degradation_max: int = 5,
    ):
        super().__init__(
            data,
            val_evaluator,
            logger=logger,
            suggest_overwrite=suggest_overwrite,
            fixed_params=fixed_params,
            max_epoch=max_epoch,
            validate_epoch=validate_epoch,
            score_degradation_max=score_degradation_max,
        )

    @classmethod
    def tune_range_given_memory_budget(
        cls, X: InteractionMatrix, memory_budget: int
    ) -> Sequence[Suggestion]:
        n_components = _get_maximal_n_components_for_budget(X, memory_budget, 300)
        return [
            IntegerSuggestion("n_components", 4, n_components),
        ]

    def optimize_doubling_dimension(
        self,
        initial_dimension: int,
        maximal_dimension: int,
        storage: Optional[optuna.storages.RDBStorage] = None,
        study_name_prefix: Optional[str] = None,
        n_trials_initial: int = 40,
        n_trials_following: int = 20,
        n_startup_trials_initial: int = 10,
        n_startup_trials_following: int = 5,
        neighborhood_scale: float = 3.0,
        suggest_overwrite_initial: Sequence[Suggestion] = [],
        random_seed: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        r"""Perform tuning gradually doubling `n_components`.
        Typically, with the initial `n_components`, the search will be more exhaustive, and with larger `n_components`, less exploration will be done around previously found parameters.
        This strategy is described in `Revisiting the Performance of iALS on Item Recommendation Benchmarks <https://arxiv.org/abs/2110.14037>`_.

        Args:
            initial_dimension: The initial dimension.
            maximal_dimension: The maximal (inclusive) dimension to be tried.
            storage:
                The storage where multiple `optuna.Study` will be created corresponding to the various dimensions.
                If `None`, all `Study` will be created in-memory.
            study_name_prefix:
                The prefix for the names of `optuna.Study`. For dimension `d`, the full name of the `Study` will be `"{study_name_prefix}_{d}"`.
                If `None`, we will use a random string for this prefix.
            n_trials_initial:
                The number of trials for the initial dimension.
            n_trials_following:
                The number of trials for the following dimensions.
            n_startup_trials_initial:
                Passed on to `n_startup_trials` argument of `optuna.pruners.MedianPruner` in the initial `optuna.Study`.
                Defaults to `10`.
            n_startup_trials_following:
                Passed on to `n_startup_trials` argument of `optuna.pruners.MedianPruner` in the following `optuna.Study`.
                Defaults to `5`.
            neighborhood_scale:
                `alpha_0` and `reg` parameters will be searched within the log-uniform range
                [`previous_dimension_result / neighborhood_scale`, `previous_dimension_result * neighborhood_scale`].
                Defaults to `3.0`
            suggest_overwrite_initial:
                Overwrites the suggestion parameters in the initial `optuna.Study`.
                Defaults to `[]`.
            random_seed:
                The random seed to control ``optuna.samplers.TPESampler``. Defaults to `None`.

        Returns:
            A tuple that consists of
                1. A dict containing the best paramaters.
                   This dict can be passed to the recommender as ``**kwargs``.
                2. A ``pandas.DataFrame`` that contains the history of optimization for all dimensions.
        """
        if study_name_prefix is None:
            study_name_prefix = str(uuid1())
        dimension = initial_dimension
        prev_params: Optional[Dict[str, float]] = None
        results: List[Tuple[float, Dict[str, Any], pd.DataFrame]] = []
        while dimension <= maximal_dimension:
            if random_seed is not None:
                random_seed += 1
            if prev_params is None:
                optimizer = self.__class__(
                    self._data,
                    self.val_evaluator,
                    suggest_overwrite=suggest_overwrite_initial,
                    fixed_params=dict(n_components=dimension),
                )
                n_trials = n_trials_initial
                n_startup_trials = n_startup_trials_initial
            else:
                suggest_overwrite = [
                    LogUniformSuggestion(
                        name, value / neighborhood_scale, value * neighborhood_scale
                    )
                    for name, value in prev_params.items()
                    if isinstance(value, float)
                ]
                optimizer = self.__class__(
                    self._data,
                    self.val_evaluator,
                    suggest_overwrite=suggest_overwrite,
                    fixed_params=dict(n_components=dimension),
                )
                n_trials = n_trials_following
                n_startup_trials = n_startup_trials_following
            self.logger.info("Search for n_components = %d", dimension)

            study = optuna.create_study(
                storage=storage,
                sampler=optuna.samplers.TPESampler(seed=random_seed),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials),
                study_name=f"{study_name_prefix}_{dimension}",
            )
            bp, df_ = optimizer.optimize_with_study(study, n_trials=n_trials)
            df_["n_components"] = dimension
            results.append((float(df_["value"].min()), bp, df_))
            prev_params = {
                key: float(value) for key, value in study.best_params.items()
            }
            dimension *= 2
        final_result_df = pd.concat([x[2] for x in results])
        final_bp = sorted(results)[0][1]

        return final_bp, final_result_df
