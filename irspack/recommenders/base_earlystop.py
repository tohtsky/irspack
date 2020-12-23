from abc import ABC, abstractmethod
from io import BytesIO
from typing import IO, Any, Optional

from optuna import Trial, exceptions
from tqdm import tqdm

from ..definitions import InteractionMatrix
from ..evaluator import Evaluator
from .base import BaseRecommender


class TrainerBase(ABC):
    def __init__(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def load_state(self, ifs: IO) -> None:
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def save_state(self, ofs: IO) -> None:
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def run_epoch(self) -> None:
        raise NotImplementedError()  # pragma: no cover


class BaseRecommenderWithEarlyStopping(BaseRecommender):
    trainer_class = TrainerBase

    def __init__(
        self,
        X_all: InteractionMatrix,
        max_epoch: int = 512,
        validate_epoch: int = 5,
        score_degration_max: int = 3,
        **kwargs: Any,
    ):
        super().__init__(X_all, **kwargs)
        self.max_epoch = max_epoch
        self.validate_epoch = validate_epoch
        if max_epoch < validate_epoch:
            raise ValueError("max_epoch must be greater than validate_epoch.")
        self.score_degradation_max = score_degration_max
        self.trainer: Optional[TrainerBase] = None
        self.best_state: Optional[bytes] = None

    @abstractmethod
    def create_trainer(self) -> TrainerBase:
        pass  # pragma: no cover

    def start_learning(self) -> None:
        self.trainer = self.create_trainer()

    def run_epoch(self) -> None:
        if self.trainer is None:
            raise RuntimeError("'run_epoch' called before initializing the trainer.")
        self.trainer.run_epoch()

    def save_state(self) -> None:
        if self.trainer is None:
            raise RuntimeError("'save_state' called before initializing the trainer.")
        with BytesIO() as ofs:
            self.trainer.save_state(ofs)
            self.best_state = ofs.getvalue()

    def load_state(self) -> None:
        if self.trainer is None:
            raise RuntimeError("'load_state' called before initializing the trainer.")
        if self.best_state is None:
            raise RuntimeError("'load_state' called before achieving any results.")
        with BytesIO(self.best_state) as ifs:
            self.trainer.load_state(ifs)

    def _learn(self) -> None:
        self.learn_with_optimizer(None, None)

    def learn_with_optimizer(
        self, evaluator: Optional[Evaluator], trial: Optional[Trial]
    ) -> None:
        self.start_learning()
        best_score = -float("inf")
        n_score_degradation = 0

        with tqdm(total=self.max_epoch) as progress_bar:
            for epoch in range(self.max_epoch):
                self.run_epoch()
                progress_bar.update(1)
                if (epoch + 1) % self.validate_epoch:
                    continue

                if evaluator is None:
                    continue

                valid_score = evaluator.get_score(self)

                progress_bar.set_description(
                    f"valid_score={valid_score[evaluator.target_metric.value]}"
                )
                relevant_score = valid_score[evaluator.target_metric.value]
                if relevant_score > best_score:
                    best_score = relevant_score
                    self.save_state()
                    self.learnt_config["max_epoch"] = epoch + 1
                    n_score_degradation = 0
                else:
                    n_score_degradation += 1
                    if n_score_degradation >= self.score_degradation_max:
                        break
                if trial is not None:
                    trial.report(-relevant_score, epoch)
                    if trial.should_prune():
                        raise exceptions.TrialPruned()

            if evaluator is not None:
                self.load_state()
