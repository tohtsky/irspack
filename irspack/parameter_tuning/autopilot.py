import re
import time
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid1

import numpy as np
import optuna
from optuna.samplers import TPESampler

from irspack.definitions import InteractionMatrix
from irspack.evaluator import Evaluator
from irspack.optimizers.base_optimizer import LowMemoryError, get_optimizer_class
from irspack.parameter_tuning.parameter_range import Suggestion, is_valid_param_name

DEFAULT_SEARCHNAMES = ["RP3beta", "IALS", "DenseSLIM", "AsymmetricCosineKNN", "SLIM"]


def search_one(
    X: InteractionMatrix,
    evaluator: Evaluator,
    optimizer_names: List[str],
    suggest_overwrites: Dict[str, List[Suggestion]],
    intermediate_result_path: Path,
    random_seed: int,
    **kwargs: Any,
) -> None:
    study = optuna.load_study(
        storage=f"sqlite:///{intermediate_result_path.name}",
        study_name="autopilot",
        sampler=TPESampler(seed=random_seed),
    )

    def objective(trial: optuna.Trial) -> float:
        optimizer_name = trial.suggest_categorical("optimizer_name", optimizer_names)
        assert isinstance(optimizer_name, str)
        optimizer = get_optimizer_class(optimizer_name)(
            X, evaluator, suggest_overwrite=suggest_overwrites[optimizer_name]
        )
        return optimizer.objective_function(optimizer_name + ".")(trial)

    study.optimize(objective, n_trials=1)


def autopilot(
    X: InteractionMatrix,
    evaluator: Evaluator,
    memory_budget: int = 4000,  # 4GB
    timeout: Optional[int] = None,
    n_trials: int = 20,
    searched_recommenders: List[str] = DEFAULT_SEARCHNAMES,
    random_seed: Optional[int] = None,
    cleanup_study: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    RNS = np.random.RandomState(random_seed)
    assert len(searched_recommenders) > 0
    suggest_overwrites: Dict[str, List[Suggestion]] = {}
    optimizer_names: List[str] = []
    db_path = Path(f".autopilot-{uuid1()}.db")
    for rec_name in searched_recommenders:
        optimizer_class_name = rec_name + "Optimizer"
        optimizer_class = get_optimizer_class(optimizer_class_name)
        try:
            suggest_overwrites[
                optimizer_class_name
            ] = optimizer_class.tune_range_given_memory_budget(X, memory_budget)
            optimizer_names.append(optimizer_class_name)
        except LowMemoryError:
            continue

    print(optimizer_names)
    study = optuna.create_study(
        storage=f"sqlite:///{db_path.name}",
        study_name="autopilot",
    )
    start = time.time()
    for _ in range(n_trials):

        timeout_for_this_process: Optional[int] = None
        if timeout is not None:
            timeout_for_this_process = timeout * 5 // n_trials
        p = Process(
            target=search_one,
            args=(
                X,
                evaluator,
                optimizer_names,
                suggest_overwrites,
                db_path,
                RNS.randint(0, 2 ** 31),
            ),
        )
        p.start()
        p.join(timeout=timeout_for_this_process)
        if p.exitcode is None:
            p.terminate()

        now = time.time()
        elapsed = now - start
        if timeout is not None:
            if elapsed > timeout:
                break
    study = optuna.load_study(
        storage=f"sqlite:///{db_path.name}",
        study_name="autopilot",
    )
    study.best_trial
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
    optimizer_name: str = best_params_with_prefix.pop("optimizer_name")
    if cleanup_study:
        db_path.unlink()
    return optimizer_name, best_params
