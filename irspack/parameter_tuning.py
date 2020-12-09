"""
Copyright 2020 BizReach, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from abc import ABC, abstractmethod
from typing import List, Any
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


def overwrite_suggestions(base: List[Suggestion], overwrite: List[Suggestion]):
    overwritten_parameter_names = {x.name for x in overwrite}
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

    def suggest(self, trial: Trial) -> float:
        return trial.suggest_uniform(self.name, self.low, self.high)


class LogUniformSuggestion(Suggestion):
    def __init__(self, name: str, low: float, high: float):
        super().__init__(name)
        if low > high:
            raise ValueError("Got low > high.")

        self.low = low
        self.high = high

    def suggest(self, trial: Trial) -> float:
        return trial.suggest_loguniform(self.name, self.low, self.high)


class IntegerSuggestion(Suggestion):
    def __init__(self, name: str, low: int, high: int, step: int = 1):
        super().__init__(name)
        if low > high:
            raise ValueError("Got low > high.")

        self.low = low
        self.high = high
        self.step = step

    def suggest(self, trial: Trial) -> int:
        return trial.suggest_int(self.name, self.low, self.high, step=self.step)


class IntegerLogUniformSuggestion(Suggestion):
    def __init__(self, name: str, low: int, high: int):
        super().__init__(name)
        if low > high:
            raise ValueError("Got low > high.")

        self.low = low
        self.high = high

    def suggest(self, trial: Trial) -> int:
        return round(trial.suggest_loguniform(self.name, self.low, self.high))


class CategoricalSuggestion(Suggestion):
    def __init__(self, name: str, choices: List[Any]):
        super().__init__(name)
        self.choices = choices

    def suggest(self, trial: Trial) -> Any:
        return trial.suggest_categorical(self.name, self.choices)


class FixedSuggestion(Suggestion):
    def __init__(self, name: str, value: Any):
        super().__init__(name)
        self.value = value

    def suggest(self, trial: Trial) -> Any:
        return self.value
