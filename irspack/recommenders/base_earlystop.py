from abc import ABC, abstractmethod
from io import BytesIO
from typing import IO, Any, Optional, Type

from optuna import Trial, exceptions
from tqdm import tqdm

from ..definitions import InteractionMatrix
from ..evaluator import Evaluator
from .base import BaseRecommender


class TrainerBase(ABC):
    """The trainer class for early-stoppable recommenders.
    The training logic of such recommenders (run-epoch, get-score) will be implemtented in the trainers."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def load_state(self, ifs: IO) -> None:
        """Load past state (with best validation score).

        Args:
            ifs (IO): The stream from which past state will be restored.
        """
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def save_state(self, ofs: IO) -> None:
        """Save the current state into stream.

        Args:
            ofs (IO): the stream to which current state will be saved.
        """
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def run_epoch(self) -> None:
        """Run single epoch."""
        raise NotImplementedError()  # pragma: no cover


class BaseRecommenderWithEarlyStopping(BaseRecommender):
    """The base class for all the early-stoppable recommenders.

    Args:
        X_train_all (csr_matrix|csc_matrix): The train interaction matrix.
        max_epoch (int, optional): The maximal number of epochs to be run. Defaults to 512.
        validate_epoch (int, optional): Frequency of validation score measurement (if any). Defaults to 5.
        score_degradation_max (int, optional): Maximal number of allowed score degradation. Defaults to 5.
    """

    trainer_class: Type[TrainerBase]

    def __init__(
        self,
        X_train_all: InteractionMatrix,
        max_epoch: int = 512,
        validate_epoch: int = 5,
        score_degradation_max: int = 5,
        **kwargs: Any,
    ):

        super().__init__(X_train_all, **kwargs)
        self.max_epoch = max_epoch
        self.validate_epoch = validate_epoch
        if max_epoch < validate_epoch:
            raise ValueError("max_epoch must be greater than validate_epoch.")
        self.score_degradation_max = score_degradation_max
        self.trainer: Optional[TrainerBase] = None
        self.best_state: Optional[bytes] = None

    @abstractmethod
    def _create_trainer(self) -> TrainerBase:
        pass  # pragma: no cover

    def start_learning(self) -> None:
        self.trainer = self._create_trainer()

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

                target_score = evaluator.get_target_score(self)

                progress_bar.set_description(f"valid_score={target_score}")
                relevant_score = target_score
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
