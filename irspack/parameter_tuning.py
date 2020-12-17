from abc import ABC, abstractmethod
from typing import List, Any, Dict
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


class Suggestion(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def suggest(self, trial: Trial) -> Any:
        pass


def overwrite_suggestions(
    base: List[Suggestion], overwrite: List[Suggestion], fixed: Dict[str, Any]
) -> List[Suggestion]:
    overwritten_parameter_names = set(
        [x.name for x in overwrite] + [x for x in fixed]
    )
    suggestions = [
        suggest_base
        for suggest_base in base
        if suggest_base.name not in overwritten_parameter_names
    ] + overwrite
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


class LogUniformSuggestion(Suggestion):
    def __init__(self, name: str, low: float, high: float):
        super().__init__(name)
        if low > high:
            raise ValueError("Got low > high.")

        self.low = low
        self.high = high

    def suggest(self, trial: Trial) -> Any:
        return trial.suggest_loguniform(self.name, self.low, self.high)


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


class IntegerLogUniformSuggestion(Suggestion):
    def __init__(self, name: str, low: int, high: int):
        super().__init__(name)
        if low > high:
            raise ValueError("Got low > high.")

        self.low = low
        self.high = high

    def suggest(self, trial: Trial) -> Any:
        return round(trial.suggest_loguniform(self.name, self.low, self.high))


class CategoricalSuggestion(Suggestion):
    def __init__(self, name: str, choices: List[Any]):
        super().__init__(name)
        self.choices = choices

    def suggest(self, trial: Trial) -> Any:
        return trial.suggest_categorical(self.name, self.choices)
