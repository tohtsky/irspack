import re
import time
import warnings
from abc import ABC, abstractmethod
from logging import Logger
from multiprocessing import Pipe as mp_pipe
from multiprocessing import Process
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from uuid import uuid1

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage
from optuna.trial import TrialState

from irspack.definitions import InteractionMatrix
from irspack.evaluator import Evaluator
from irspack.optimizers.base_optimizer import (
    LowMemoryError,
    get_optimizer_class,
    study_to_dataframe,
)
from irspack.parameter_tuning.parameter_range import Suggestion, is_valid_param_name
from irspack.recommenders import BaseRecommender
from irspack.utils.default_logger import get_default_logger

DEFAULT_SEARCHNAMES = ["RP3beta", "IALS", "DenseSLIM", "AsymmetricCosineKNN", "SLIM"]

_sort_intermediate: Callable[[Tuple[int, float]], float] = lambda _: _[1]


def search_one(
    pipe: Any,
    X: InteractionMatrix,
    evaluator: Evaluator,
    optimizer_names: List[str],
    suggest_overwrites: Dict[str, List[Suggestion]],
    db_url: str,
    study_name: str,
    random_seed: int,
    logger: Logger,
) -> None:
    study = optuna.load_study(
        storage=db_url,
        study_name=study_name,
        sampler=TPESampler(seed=random_seed),
    )

    def _obj(trial: optuna.Trial) -> float:
        pipe.send(trial.number)
        optimizer_name = trial.suggest_categorical("optimizer_name", optimizer_names)
        assert isinstance(optimizer_name, str)

        optimizer = get_optimizer_class(optimizer_name)(
            X,
            evaluator,
            suggest_overwrite=suggest_overwrites[optimizer_name],
            logger=logger,
        )
        return optimizer.objective_function(optimizer_name + ".")(trial)

    study.optimize(_obj, n_trials=1)


class TaskBackend(ABC):
    @abstractmethod
    def __init__(
        self,
        X: InteractionMatrix,
        evaluator: Evaluator,
        optimizer_names: List[str],
        suggest_overwrites: Dict[str, List[Suggestion]],
        db_url: str,
        study_name: str,
        random_seed: int,
        logger: Logger,
    ):
        raise NotImplementedError()

    @property
    def exit_code(self) -> Optional[int]:
        return self._exit_code()

    @abstractmethod
    def receive_trial_number(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def _exit_code(self) -> Optional[int]:
        raise NotImplementedError()

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def join(self, timeout: Optional[int]) -> None:
        raise NotImplementedError()

    @abstractmethod
    def terminate(self) -> None:
        raise NotImplementedError()  # pragma: no cover


class MultiProcessingBackend(TaskBackend):
    def __init__(
        self,
        X: InteractionMatrix,
        evaluator: Evaluator,
        optimizer_names: List[str],
        suggest_overwrites: Dict[str, List[Suggestion]],
        db_url: str,
        study_name: str,
        random_seed: int,
        logger: Logger,
    ):
        self.pipe_parent, pipe_child = mp_pipe()
        self._p = Process(
            target=search_one,
            args=(
                pipe_child,
                X,
                evaluator,
                optimizer_names,
                suggest_overwrites,
                db_url,
                study_name,
                random_seed,
                logger,
            ),
        )

    def _exit_code(self) -> Optional[int]:
        return self._p.exitcode

    def receive_trial_number(self) -> int:
        result: int = self.pipe_parent.recv()
        return result

    def start(self) -> None:
        self._p.start()

    def join(self, timeout: Optional[int]) -> None:
        self._p.join(timeout=timeout)

    def terminate(self) -> None:
        self._p.terminate()


class SameThreadBackend(TaskBackend):
    def __init__(
        self,
        X: InteractionMatrix,
        evaluator: Evaluator,
        optimizer_names: List[str],
        suggest_overwrites: Dict[str, List[Suggestion]],
        db_url: str,
        study_name: str,
        random_seed: int,
        logger: Logger,
    ):
        self._args = (
            X,
            evaluator,
            optimizer_names,
            suggest_overwrites,
            db_url,
            study_name,
            random_seed,
            logger,
        )
        self._trial_number: Optional[int] = None

    def send(self, i: int) -> None:
        self._trial_number = i

    def start(self) -> None:
        search_one(self, *self._args)

    def receive_trial_number(self) -> int:
        assert self._trial_number is not None
        return self._trial_number

    def join(self, timeout: Optional[int]) -> None:
        warnings.warn("Single step timeout will be ignored for ThreadingBackend.")
        pass

    def terminate(self) -> None:
        # can't do that
        pass

    def _exit_code(self) -> Optional[int]:
        # must always normally exit
        return 0


def autopilot(
    X: InteractionMatrix,
    evaluator: Evaluator,
    n_trials: int = 20,
    memory_budget: int = 4000,  # 4GB
    timeout_overall: Optional[int] = None,
    timeout_singlestep: Optional[int] = None,
    algorithms: List[str] = DEFAULT_SEARCHNAMES,
    random_seed: Optional[int] = None,
    logger: Optional[Logger] = None,
    callback: Optional[Callable[[int, pd.DataFrame], None]] = None,
    storage: Optional[RDBStorage] = None,
    study_name: Optional[str] = None,
    task_resource_provider: Type[TaskBackend] = MultiProcessingBackend,
) -> Tuple[Type[BaseRecommender], Dict[str, Any], pd.DataFrame]:

    r"""Given an interaction matrix and an evaluator, search for the best algorithm and its parameters
    (roughly) within the time & space constraints. You can specify how each search step will be executed.

    Args:
        X:
            Input interaction matrix.
        evaluator:
            Evaluator to measure the performance of the recommenders.
        n_trials: The maximal number of trials. Defaults to 20.
        memory_budget:
            Optimizers will try search parameters so that memory usage (in megabyte) will not exceed this values.
            An algorithm will not be searched if it inevitably violates this bound.
            Note that this value is quite rough one and will not be respected strictly.
        timeout_overall:
            If set, the total execution time of the trials will not exceed this value (roughly).
        timeout_singlestep:
            If set, a single trial (recommender and a set of its parameter) will not run for more than the value (in seconds).
            Such a trial is considered to have produced  a score value of 0,
            and optuna will avoid suggesting such values (if everything works fine).
            Defaults to `None`.
        algorithms:
            A list of algorithm names to be tried.
            Defaults to `["RP3beta", "IALS", "DenseSLIM", "AsymmetricCosineKNN", "SLIM"]`.
        random_seed:
            The random seed that controls the suggestion behavior.
            Defaults to `None`.
        logger:
            The logger to be used. If `None`, irspack's default logger will be used.
            Defaults to None.
        callback:
            If not `None`, called at the end of every single trial with the following arguments

                1. The current trial's number.
                2. A `pd.DataFrame` that holds history of trial execution.

            Defaults to `None`.
        storage:
            An instance of `optuna.storages.RDBStorage`. Defaults to `None`.
        study_name:
            If `storage` argument is given, you have to pass study_name
            argument.
        task_resource_provider:
            Specifies how each search step is executed. Defaults to `MultiProceesingBackend`.
    Raises:
        ValueError:
            If `storage` is given but `study_name` is not specified.
        RuntimeError:
            If no recommender algorithms are available within given memory budget.
        RuntimeError:
            If no trials have been completed within given timeout.


    Returns:

        * The best algorithm's recommender class.
        * The best parameters.
        * The dataframe containing the history of trials.

    """
    if storage is not None and study_name is None:
        raise ValueError('"study_name" must be specified if "storage" is given.')
    RNS = np.random.RandomState(random_seed)
    suggest_overwrites: Dict[str, List[Suggestion]] = {}
    optimizer_names: List[str] = []
    for rec_name in algorithms:
        optimizer_class_name = rec_name + "Optimizer"
        optimizer_class = get_optimizer_class(optimizer_class_name)
        try:
            suggest_overwrites[
                optimizer_class_name
            ] = optimizer_class.tune_range_given_memory_budget(X, memory_budget)
            optimizer_names.append(optimizer_class_name)
        except LowMemoryError:
            continue

    if not optimizer_names:
        raise RuntimeError("No available algorithm with given memory.")

    if logger is None:
        logger = get_default_logger()

    logger.info("Trying the following algorithms: %s", optimizer_names)

    optional_db_path = Path(f".autopilot-{uuid1()}.db")
    storage_: RDBStorage
    if storage is None:
        storage_ = RDBStorage(
            url=f"sqlite:///{optional_db_path.name}",
        )
    else:
        storage_ = storage

    if study_name is None:
        study_name_ = f"autopilot-{uuid1()}"
    else:
        study_name_ = study_name
    start = time.time()
    study = optuna.create_study(
        storage=storage_, study_name=study_name_, load_if_exists=True
    )
    study_id = storage_.get_study_id_from_name(study_name_)

    for _ in range(n_trials):

        task_start = time.time()
        elapsed_at_start = task_start - start

        timeout_for_this_process: Optional[int] = None
        if timeout_overall is None:
            timeout_for_this_process = timeout_singlestep
        else:
            remaining_time = int(timeout_overall - elapsed_at_start)
            if remaining_time <= 0:
                break

            if timeout_singlestep is not None:
                timeout_for_this_process = min(remaining_time, timeout_singlestep)
        task = task_resource_provider(
            X,
            evaluator,
            optimizer_names,
            suggest_overwrites,
            storage_.url,
            study_name_,
            RNS.randint(0, np.iinfo(np.int32).max, dtype=np.int32),
            logger,
        )

        task.start()
        trial_number = task.receive_trial_number()
        task.join(timeout=timeout_for_this_process)

        if task.exit_code is None:
            task.terminate()
            try:
                logger.info(f"Trial {trial_number} timeout.")
                storage_.read_trials_from_remote_storage(study_id)
                trial_id = storage_.get_trial_id_from_study_id_trial_number(
                    study_id, trial_number
                )
                trial_this = storage_.get_trial(trial_id)
                intermediate_values = sorted(
                    list(trial_this.intermediate_values.items()),
                    key=_sort_intermediate,
                )

                if intermediate_values:
                    # Though terminated, it resulted in some values.
                    # Regard it as a COMPLETE trial.
                    storage_.set_trial_values(
                        trial_id,
                        [intermediate_values[0][1]],
                    )
                    storage_.set_trial_user_attr(
                        trial_id, "max_epoch", intermediate_values[0][0] + 1
                    )
                else:
                    # Penalize such a time-consuming trial
                    storage_.set_trial_values(trial_id, [0.0])
                storage_.set_trial_state(trial_id, TrialState.COMPLETE)
            except RuntimeError:  # pragma: no cover
                pass  # pragma: no cover

        if callback is not None:
            callback(trial_number, study_to_dataframe(study))

        now = time.time()
        elapsed = now - start
        if timeout_overall is not None:
            if elapsed > timeout_overall:
                break
    best_params_with_prefix = dict(
        **study.best_trial.params,
        **{
            key: val
            for key, val in study.best_trial.user_attrs.items()
            if is_valid_param_name(key)
        },
    )
    best_params = {
        re.sub(r"^([^\.]*\.)", "", key): value
        for key, value in best_params_with_prefix.items()
    }
    optimizer_name: str = best_params.pop("optimizer_name")
    result_df = study_to_dataframe(study)

    if storage is None:
        optional_db_path.unlink()
    recommender_class = get_optimizer_class(optimizer_name).recommender_class

    return (recommender_class, best_params, result_df)
