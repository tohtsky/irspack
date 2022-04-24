import re
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Sequence

from optuna import Trial

NAME_CHECKER = re.compile(r"^([a-zA-Z\d]+[\-_]*)+$")


def is_valid_param_name(name: str) -> bool:
    if NAME_CHECKER.match(name) is None:
        return False
    return True


class Suggestion(object, metaclass=ABCMeta):
    def __init__(self, name: str):
        """The base class to controll optuna's ``Trial`` behavior during
            hyper parameter optimization.

        Args:
            name (str): The name of the parameter to be tuned.
        """
        if not is_valid_param_name(name):
            raise ValueError(
                rf""""{name}" is  not a valid parameter name. It should match r"^([a-zA-Z\d]+[\-_]*)+$"""
            )
        self.name = name

    @abstractmethod
    def suggest(self, trial: Trial, prefix: str = "") -> Any:
        raise NotImplementedError('"suggest" must be implemented.')

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError('"__repr__" must be implemented.')


def overwrite_suggestions(
    base: Sequence[Suggestion],
    suggest_overwrite: Sequence[Suggestion],
    fixed_params: Dict[str, Any],
) -> Sequence[Suggestion]:
    for suggest in suggest_overwrite:
        if suggest.name in fixed_params:
            raise ValueError("suggest_overwrite and fixed_param have overwrap.")

    overwritten_parameter_names = set(
        [x.name for x in suggest_overwrite] + [x for x in fixed_params]
    )
    suggestions = [
        suggest_base
        for suggest_base in base
        if suggest_base.name not in overwritten_parameter_names
    ] + list(suggest_overwrite)
    return suggestions


class UniformSuggestion(Suggestion):
    def __init__(self, name: str, low: float, high: float):
        super().__init__(name)
        if low > high:
            raise ValueError("Got low > high.")

        self.low = low
        self.high = high

    def suggest(self, trial: Trial, prefix: str = "") -> Any:
        return trial.suggest_uniform(prefix + self.name, self.low, self.high)

    def __repr__(self) -> str:
        return f"UniformSuggestion(name={self.name!r}, low={self.low!r}, high={self.high!r})"


class LogUniformSuggestion(Suggestion):
    def __init__(self, name: str, low: float, high: float):
        super().__init__(name)
        if low > high:
            raise ValueError("Got low > high.")

        self.low = low
        self.high = high

    def suggest(self, trial: Trial, prefix: str = "") -> Any:
        return trial.suggest_loguniform(prefix + self.name, self.low, self.high)

    def __repr__(self) -> str:
        return f"LogUniformSuggestion(name={self.name!r}, low={self.low!r}, high={self.high!r})"


class IntegerSuggestion(Suggestion):
    def __init__(self, name: str, low: int, high: int, step: int = 1):
        super().__init__(name)
        if low > high:
            raise ValueError("Got low > high.")

        self.low = low
        self.high = high
        self.step = step

    def suggest(self, trial: Trial, prefix: str = "") -> Any:
        return trial.suggest_int(
            prefix + self.name, self.low, self.high, step=self.step
        )

    def __repr__(self) -> str:
        return f"IntegerSuggestion(name={self.name!r}, low={self.low!r}, high={self.high!r})"


class IntegerLogUniformSuggestion(Suggestion):
    def __init__(self, name: str, low: int, high: int):
        super().__init__(name)
        if low > high:
            raise ValueError("Got low > high.")

        self.low = low
        self.high = high

    def suggest(self, trial: Trial, prefix: str = "") -> Any:
        return round(trial.suggest_loguniform(prefix + self.name, self.low, self.high))

    def __repr__(self) -> str:
        return f"IntegerLogUniformSuggestion(name={self.name!r}, low={self.low!r}, high={self.high!r})"


class CategoricalSuggestion(Suggestion):
    def __init__(self, name: str, choices: List[Any]):
        super().__init__(name)
        self.choices = choices

    def suggest(self, trial: Trial, prefix: str = "") -> Any:
        return trial.suggest_categorical(prefix + self.name, self.choices)

    def __repr__(self) -> str:
        return (
            f"IntegerLogUniformSuggestion(name={self.name!r}, choices={self.choices!r})"
        )
