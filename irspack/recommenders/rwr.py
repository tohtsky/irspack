from irspack.utils import get_n_threads

from ..definitions import InteractionMatrix
from ._rwr import RandomWalkGenerator
from .base import BaseSimilarityRecommender


class RandomWalkWithRestartRecommender(BaseSimilarityRecommender):
    def __init__(
        self,
        X_train_all: InteractionMatrix,
        decay: float = 0.3,
        cutoff: int = 1000,
        n_samples: int = 1000,
        random_seed: int = 42,
        n_threads: int = 4,
    ):
        super().__init__(X_train_all)
        self.decay = decay
        self.n_samples = n_samples
        self.cutoff = cutoff
        self.random_seed = random_seed
        self.n_threads = get_n_threads(n_threads)

    def _learn(self) -> None:
        rwg = RandomWalkGenerator(self.X_train_all.tocsr())
        self.W_ = rwg.run_with_restart(
            self.decay,
            self.cutoff,
            self.n_samples,
            self.n_threads,
            self.random_seed,
        )
        self.W_ = self.W_.tocsc() / self.n_samples
