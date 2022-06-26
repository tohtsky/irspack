import logging
import re
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type

import pandas as pd

from ..default_logger import get_default_logger
from .parameter_range import is_valid_param_name

if TYPE_CHECKING:
    from optuna import Study, Trial

    from ..evaluation import Evaluator
    from ..recommenders.base import BaseRecommender, InteractionMatrix

SparseMatrixSuggestFunction = Callable[["Trial"], "InteractionMatrix"]
ParameterSuggestFunction = Callable[["Trial"], Dict[str, Any]]


def add_score_to_trial(trial: "Trial", score: Dict[str, float], cutoff: int) -> None:
    score_history: List[Tuple[int, Dict[str, float]]] = trial.study.user_attrs.get(
        "scores", []
    )
    score_history.append(
        (
            trial.number,
            {
                f"{score_name}@{cutoff}": score_value
                for score_name, score_value in score.items()
            },
        )
    )
    trial.study.set_user_attr("scores", score_history)


def study_to_dataframe(study: "Study") -> pd.DataFrame:
    result_df: pd.DataFrame = study.trials_dataframe().set_index("number")

    # remove prefix
    result_df.columns = [
        re.sub(r"^(user_attrs|params)_", "", colname) for colname in result_df.columns
    ]

    trial_and_scores: List[Tuple[float, Dict[str, float]]] = study.user_attrs.get(
        "scores", []
    )
    score_df = pd.DataFrame(
        [x[1] for x in trial_and_scores],
        index=[x[0] for x in trial_and_scores],
    )
    score_df.index.name = "number"
    result_df = result_df.join(score_df, how="left")
    return result_df


class Optimizer:
    r"""
    Args:
        data_suggest_function:
            The train data.
        val_evaluator:
            The validation evaluator that measures the performance of the recommenders.
        logger:
            The logger used during the optimization steps. Defaults to `None`.
            If `None`, the default logger of irspack will be used.
        suggest_overwrite:
            Customizes (e.g. enlarging the parameter region or adding new parameters to be tuned)
            the default parameter search space defined by ``default_tune_range``
            Defaults to list().
        fixed_params (Dict[str, Any], optional):
            Fixed parameters passed to recommenders during the optimization procedure.
            If such a parameter exists in ``default_tune_range``, it will not be tuned.
            Defaults to dict().
        max_epoch (int, optional):
            The maximal number of epochs for the training. Defaults to 512.
        validate_epoch (int, optional):
            The frequency of validation score measurement. Defaults to 5.
        score_degradation_max (int, optional):
            Maximal number of allowed score degradation. Defaults to 5. Defaults to 5."""

    def __init__(
        self,
        data_suggest_function: SparseMatrixSuggestFunction,
        parameter_suggest_function: ParameterSuggestFunction,
        fixed_params: Dict[str, Any],
        val_evaluator: "Evaluator",
        logger: Optional[logging.Logger] = None,
        max_epoch: int = 128,
        validate_epoch: int = 5,
        score_degradation_max: int = 5,
    ):

        if logger is None:

            logger = get_default_logger()

        self.logger = logger
        self._data_suggest_function = data_suggest_function
        self._parameter_suggest_function = parameter_suggest_function
        self.val_evaluator = val_evaluator

        self.current_trial: int = 0
        self.best_val = float("inf")
        self.fixed_params = fixed_params

        self.max_epoch = max_epoch
        self.validate_epoch = validate_epoch
        self.score_degradation_max = score_degradation_max

    def objective_function(
        self, recommender_class: Type["BaseRecommender"]
    ) -> Callable[["Trial"], float]:
        r"""Returns the objective function that can be passed to ``optuna.Study`` .

        Returns:
            A callable that receives ``otpuna.Trial`` and returns float (like ndcg score).
        """

        def objective_func(trial: "Trial") -> float:
            start = time.time()
            data = self._data_suggest_function(trial)
            params = self._parameter_suggest_function(trial)
            params.update(self.fixed_params)
            self.logger.info("Trial %s:", trial.number)
            self.logger.info("parameter = %s", params)

            recommender = recommender_class(data, **params)
            recommender.learn_with_optimizer(
                self.val_evaluator,
                trial,
                max_epoch=self.max_epoch,
                validate_epoch=self.validate_epoch,
                score_degradation_max=self.score_degradation_max,
            )

            score = self.val_evaluator.get_score(recommender)
            end = time.time()

            time_spent = end - start
            self.logger.info(
                "Config %d obtained the following scores: %s within %f seconds.",
                trial.number,
                score,
                time_spent,
            )
            val_score = score[self.val_evaluator.target_metric.name]

            # max_epoch will be stored in learnt_config
            for param_name, param_val in recommender.learnt_config.items():
                trial.set_user_attr(param_name, param_val)

            add_score_to_trial(trial, score, self.val_evaluator.cutoff)

            return -val_score

        return objective_func

    def optimize_with_study(
        self,
        study: "Study",
        recommender_class: Type["BaseRecommender"],
        n_trials: int = 20,
        timeout: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Perform the optimization step using the user-created ``optuna.Study`` object.
        Creating and managing the study object will be convenient e.g. when you

            1. want to `store/resume the study using RDB backend <https://optuna.readthedocs.io/en/stable/tutorial/003_rdb.html>`_.
            2. want perform a `distributed optimization <https://optuna.readthedocs.io/en/stable/tutorial/004_distributed.html>`_.

        Args:
            study:
                The study object.
            n_trials:
                The number of expected trials (include pruned trial.). Defaults to 20.
        Returns:
            A tuple that consists of

                1. A dict containing the best paramaters.
                   This dict can be passed to the recommender as ``**kwargs``.
                2. A ``pandas.DataFrame`` that contains the history of optimization.

        """

        objective_func = self.objective_function(recommender_class)

        self.logger.info(
            """Start parameter search for %s""",
            recommender_class.__name__,
        )

        study.optimize(objective_func, n_trials=n_trials, timeout=timeout)
        best_params = dict(
            **study.best_trial.params,
            **{
                key: val
                for key, val in study.best_trial.user_attrs.items()
                if is_valid_param_name(key)
            },
        )
        best_params.update(self.fixed_params)
        trials_df = study_to_dataframe(study)
        return (best_params, trials_df)
