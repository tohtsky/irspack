from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List

from optuna import Trial

__all__ = [
    "Suggestion",
    "UniformSuggestion",
    "LogUniformSuggestion",
    "IntegerSuggestion",
    "IntegerLogUniformSuggestion",
    "CategoricalSuggestion",
    "overwrite_suggestions",
]


class Suggestion(object, metaclass=ABCMeta):
    def __init__(self, name: str):
        """The base class to controll optuna's ``Trial`` behavior during
            hyper parameter optimization.

        Args:
            name (str): The name of the parameter to be tuned.
        """
        self.name = name

    @abstractmethod
    def suggest(self, trial: Trial) -> Any:
        raise NotImplementedError('"suggest" must be implemented.')

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError('"__repr__" must be implemented.')


def overwrite_suggestions(
    base: List[Suggestion],
    suggest_overwrite: List[Suggestion],
    fixed_params: Dict[str, Any],
) -> List[Suggestion]:
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
    ] + suggest_overwrite
    return suggestions


class UniformSuggestion(Suggestion):
    def __init__(self, name: str, low: float, high: float):
        super().__init__(name)
        if low > high:
            raise ValueError("Got low > high.")

        self.low = low
        self.high = high

    def suggest(self, trial: Trial) -> Any:
        return trial.suggest_uniform(self.name, self.low, self.high)

    def __repr__(self) -> str:
        return f"UniformSuggestion(name={self.name!r}, low={self.low!r}, high={self.high!r})"


class LogUniformSuggestion(Suggestion):
    def __init__(self, name: str, low: float, high: float):
        super().__init__(name)
        if low > high:
            raise ValueError("Got low > high.")

        self.low = low
        self.high = high

    def suggest(self, trial: Trial) -> Any:
        return trial.suggest_loguniform(self.name, self.low, self.high)

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

    def suggest(self, trial: Trial) -> Any:
        return trial.suggest_int(self.name, self.low, self.high, step=self.step)

    def __repr__(self) -> str:
        return f"IntegerSuggestion(name={self.name!r}, low={self.low!r}, high={self.high!r})"


class IntegerLogUniformSuggestion(Suggestion):
    def __init__(self, name: str, low: int, high: int):
        super().__init__(name)
        if low > high:
            raise ValueError("Got low > high.")

        self.low = low
        self.high = high

    def suggest(self, trial: Trial) -> Any:
        return round(trial.suggest_loguniform(self.name, self.low, self.high))

    def __repr__(self) -> str:
        return f"IntegerLogUniformSuggestion(name={self.name!r}, low={self.low!r}, high={self.high!r})"


class CategoricalSuggestion(Suggestion):
    def __init__(self, name: str, choices: List[Any]):
        super().__init__(name)
        self.choices = choices

    def suggest(self, trial: Trial) -> Any:
        return trial.suggest_categorical(self.name, self.choices)

    def __repr__(self) -> str:
        return (
            f"IntegerLogUniformSuggestion(name={self.name!r}, choices={self.choices!r})"
        )
