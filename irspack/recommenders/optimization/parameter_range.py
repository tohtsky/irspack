import re
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from optuna import Trial

NAME_CHECKER = re.compile(r"^([a-zA-Z\d]+[\-_]*)+$")


def is_valid_param_name(name: str) -> bool:
    if NAME_CHECKER.match(name) is None:
        return False
    return True


class ParameterRange(metaclass=ABCMeta):
    r"""The class to define default parameter tuning range.

    Args:
        name (str): The name of the parameter to be tuned.
    """

    def __init__(self, name: str):
        if not is_valid_param_name(name):
            raise ValueError(
                rf""""{name}" is  not a valid parameter name. It should match r"^([a-zA-Z\d]+[\-_]*)+$"""
            )
        self.name = name

    @abstractmethod
    def suggest(self, trial: "Trial", prefix: str = "") -> Any:
        raise NotImplementedError('"suggest" must be implemented.')  # pragma: no cover


class UniformFloatRange(ParameterRange):
    def __init__(self, name: str, low: float, high: float):
        super().__init__(name)
        if low > high:
            raise ValueError("Got low > high.")

        self.low = low
        self.high = high

    def suggest(self, trial: "Trial", prefix: str = "") -> Any:
        return trial.suggest_float(prefix + self.name, self.low, self.high)


class LogUniformFloatRange(ParameterRange):
    def __init__(self, name: str, low: float, high: float):
        super().__init__(name)
        if low > high:
            raise ValueError("Got low > high.")

        self.low = low
        self.high = high

    def suggest(self, trial: "Trial", prefix: str = "") -> Any:
        return trial.suggest_float(prefix + self.name, self.low, self.high, log=True)


class UniformIntegerRange(ParameterRange):
    def __init__(self, name: str, low: int, high: int, step: int = 1):
        super().__init__(name)
        if low > high:
            raise ValueError("Got low > high.")

        self.low = low
        self.high = high
        self.step = step

    def suggest(self, trial: "Trial", prefix: str = "") -> Any:
        return trial.suggest_int(
            prefix + self.name, self.low, self.high, step=self.step
        )


class LogUniformIntegerRange(ParameterRange):
    def __init__(self, name: str, low: int, high: int):
        super().__init__(name)
        if low > high:
            raise ValueError("Got low > high.")

        self.low = low
        self.high = high

    def suggest(self, trial: "Trial", prefix: str = "") -> Any:
        return trial.suggest_int(prefix + self.name, self.low, self.high, log=True)


class CategoricalRange(ParameterRange):
    def __init__(self, name: str, choices: List[Any]):
        super().__init__(name)
        self.choices = choices

    def suggest(self, trial: "Trial", prefix: str = "") -> Any:
        return trial.suggest_categorical(prefix + self.name, self.choices)


default_tune_range_knn = [
    UniformIntegerRange("top_k", 4, 1000),
    UniformFloatRange("shrinkage", 0, 1000),
]

default_tune_range_knn_with_weighting = [
    UniformIntegerRange("top_k", 4, 1000),
    UniformFloatRange("shrinkage", 0, 1000),
    CategoricalRange("feature_weighting", ["NONE", "TF_IDF", "BM_25"]),
]
