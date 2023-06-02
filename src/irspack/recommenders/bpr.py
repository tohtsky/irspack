import pickle
from typing import IO, Optional

import numpy as np
from lightfm import LightFM

from .._threading import get_n_threads
from ..definitions import (
    DenseMatrix,
    DenseScoreArray,
    InteractionMatrix,
    UserIndexArray,
)
from ..optimization import CategoricalRange, LogUniformFloatRange, UniformIntegerRange
from .base import BaseRecommenderWithItemEmbedding, BaseRecommenderWithUserEmbedding
from .base_earlystop import (
    BaseEarlyStoppingRecommenderConfig,
    BaseRecommenderWithEarlyStopping,
    TrainerBase,
)


class BPRFMTrainer(TrainerBase):
    def __init__(
        self,
        X: InteractionMatrix,
        n_components: int,
        item_alpha: float,
        user_alpha: float,
        loss: str,
        n_threads: int,
    ):
        self.X = X.tocoo()
        self.n_threads = n_threads
        self.fm = LightFM(
            no_components=n_components,
            item_alpha=item_alpha,
            user_alpha=user_alpha,
            loss=loss,
        )

    def run_epoch(self) -> None:
        self.fm.fit_partial(self.X, num_threads=self.n_threads)

    def load_state(self, ifs: IO) -> None:
        self.fm = pickle.load(ifs)

    def save_state(self, ofs: IO) -> None:
        pickle.dump(self.fm, ofs, protocol=pickle.HIGHEST_PROTOCOL)


class BPRFMConfig(BaseEarlyStoppingRecommenderConfig):
    n_components: int = 128
    item_alpha: float = 1e-9
    user_alpha: float = 1e-9
    loss: str = "bpr"
    n_threads: Optional[int] = None


class BPRFMRecommender(
    BaseRecommenderWithEarlyStopping,
    BaseRecommenderWithUserEmbedding,
    BaseRecommenderWithItemEmbedding,
):
    r"""A `LightFM <https://github.com/lyst/lightfm>`_ wrapper for our interface.

    This will create ``LightFM`` instance by

    .. code-block:: python

        fm = LightFM(
            no_components=n_components,
            item_alpha=item_alpha,
            user_alpha=user_alpha,
            loss=loss,
        )

    and run ``fm.fit_partial(X, num_threads=self.n_threads)`` to train through a single epoch.


    Args:
        X_train_all:
            Input interaction matrix.
        n_components:
            The dimension for latent factor. Defaults to 128.
        item_alpha:
            The regularization coefficient for item factors. Defaults to 1e-9.
        user_alpha:
            The regularization coefficient for user factors. Defaults to 1e-9.
        loss:
            Specifies the loss function type of LightFM. Must be one of {"bpr", "warp"}. Defaults to "bpr".
        train_epochs:
            Number of training epochs. Defaults to 128.
        n_threads: Specifies the number of threads to use for the computation.
            If ``None``, the environment variable ``"IRSPACK_NUM_THREADS_DEFAULT"`` will be looked up,
            and if the variable is not set, it will be set to ``os.cpu_count()``. Defaults to None.
    """

    config_class = BPRFMConfig
    default_tune_range = [
        UniformIntegerRange("n_components", 4, 256),
        LogUniformFloatRange("item_alpha", 1e-9, 1e-2),
        LogUniformFloatRange("user_alpha", 1e-9, 1e-2),
        CategoricalRange("loss", ["bpr", "warp"]),
    ]

    def __init__(
        self,
        X_train_all: InteractionMatrix,
        n_components: int = 128,
        item_alpha: float = 1e-9,
        user_alpha: float = 1e-9,
        loss: str = "bpr",
        n_threads: Optional[int] = None,
        train_epochs: int = 128,
    ):
        super().__init__(
            X_train_all,
            train_epochs=train_epochs,
        )
        self.n_components = n_components
        self.item_alpha = item_alpha
        self.user_alpha = user_alpha
        if loss not in {"bpr", "warp"}:
            raise ValueError('BPRFM loss must be either "bpr" or "warp".')
        self.loss = loss
        self.trainer: Optional[BPRFMTrainer] = None
        self.n_threads = get_n_threads(n_threads)

    def _create_trainer(self) -> BPRFMTrainer:
        return BPRFMTrainer(
            self.X_train_all,
            self.n_components,
            self.item_alpha,
            self.user_alpha,
            self.loss,
            self.n_threads,
        )

    @property
    def fm(self) -> LightFM:
        if self.trainer is None:
            raise RuntimeError("tried to fetch fm instance before the fit.")
        return self.trainer.fm

    def get_score(self, index: UserIndexArray) -> np.ndarray:
        res: np.ndarray = (
            self.fm.user_embeddings[index].dot(self.fm.item_embeddings.T)
            + self.fm.item_biases[np.newaxis, :]
        )
        return res

    def get_score_block(self, begin: int, end: int) -> np.ndarray:
        res: np.ndarray = (
            self.fm.user_embeddings[begin:end].dot(self.fm.item_embeddings.T)
            + self.fm.item_biases[np.newaxis, :]
        )
        return res

    def get_user_embedding(self) -> DenseMatrix:
        res: DenseMatrix = self.fm.user_embeddings
        return res

    def get_score_from_user_embedding(
        self, user_embedding: DenseMatrix
    ) -> DenseScoreArray:
        res: DenseScoreArray = (
            user_embedding.dot(self.fm.item_embeddings.T)
            + self.fm.item_biases[np.newaxis, :]
        )
        return res

    def get_item_embedding(self) -> DenseMatrix:
        res: DenseMatrix = self.fm.item_embeddings
        return res

    def get_score_from_item_embedding(
        self, user_indices: UserIndexArray, item_embedding: DenseMatrix
    ) -> DenseScoreArray:
        # ignore bias
        res: DenseScoreArray = self.fm.user_embeddings[user_indices].dot(
            item_embedding.T
        )
        return res
