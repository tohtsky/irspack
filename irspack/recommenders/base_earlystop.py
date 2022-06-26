from abc import ABC, abstractmethod
from io import BytesIO
from typing import IO, TYPE_CHECKING, Any, Optional, Type

from fastprogress import progress_bar

from ..definitions import InteractionMatrix
from .base import BaseRecommender, RecommenderConfig

if TYPE_CHECKING:
    from optuna import Trial

    from irspack import evaluation


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


class BaseEarlyStoppingRecommenderConfig(RecommenderConfig):
    train_epochs: int = 512


class BaseRecommenderWithEarlyStopping(BaseRecommender):
    """The base class for all the early-stoppable recommenders.

    Args:
        X_train_all:
            The train interaction matrix.
        train_epochs:
            The number of training epochs to run. Defaults to 128.
    """

    trainer_class: Type[TrainerBase]

    def __init__(
        self,
        X_train_all: InteractionMatrix,
        train_epochs: int = 128,
        **kwargs: Any,
    ):

        super().__init__(X_train_all, **kwargs)
        self.train_epochs = train_epochs
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
        self.learn_with_optimizer(None, None, max_epoch=self.train_epochs)

    def learn_with_optimizer(
        self,
        evaluator: Optional["evaluation.Evaluator"],
        trial: Optional["Trial"],
        max_epoch: int = 128,
        validate_epoch: int = 5,
        score_degradation_max: int = 5,
    ) -> None:
        from optuna.exceptions import TrialPruned

        self.start_learning()
        best_score = -float("inf")
        n_score_degradation = 0
        pb = progress_bar(range(max_epoch))
        for epoch in pb:
            self.run_epoch()
            if (epoch + 1) % validate_epoch:
                continue

            if evaluator is None:
                continue

            target_score = evaluator.get_target_score(self)

            pb.comment = f"{evaluator.target_metric_name}={target_score}"
            relevant_score = target_score
            if relevant_score > best_score:
                best_score = relevant_score
                self.save_state()
                self.learnt_config["train_epochs"] = epoch + 1
                n_score_degradation = 0
            else:
                n_score_degradation += 1
                if n_score_degradation >= score_degradation_max:
                    pb.on_interrupt()
                    break
            if trial is not None:
                trial.report(-relevant_score, epoch)
                if trial.should_prune():
                    pb.on_interrupt()
                    raise TrialPruned()

        if evaluator is not None:
            self.load_state()
