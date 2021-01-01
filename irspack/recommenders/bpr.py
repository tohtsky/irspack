import pickle
from typing import IO, Optional

import numpy as np
from lightfm import LightFM

from ..definitions import (
    DenseMatrix,
    DenseScoreArray,
    InteractionMatrix,
    UserIndexArray,
)
from .base import (
    BaseRecommenderWithItemEmbedding,
    BaseRecommenderWithThreadingSupport,
    BaseRecommenderWithUserEmbedding,
)
from .base_earlystop import BaseRecommenderWithEarlyStopping, TrainerBase


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


class BPRFMRecommender(
    BaseRecommenderWithEarlyStopping,
    BaseRecommenderWithThreadingSupport,
    BaseRecommenderWithUserEmbedding,
    BaseRecommenderWithItemEmbedding,
):
    def __init__(
        self,
        X_train_all: InteractionMatrix,
        n_components: int = 128,
        item_alpha: float = 1e-9,
        user_alpha: float = 1e-9,
        loss: str = "bpr",
        max_epoch: int = 512,
        n_threads: Optional[int] = None,
        validate_epoch: int = 5,
        score_degradation_max: int = 3,
    ):
        super().__init__(
            X_train_all,
            n_threads=n_threads,
            max_epoch=max_epoch,
            validate_epoch=validate_epoch,
            score_degradation_max=score_degradation_max,
        )
        self.n_components = n_components
        self.item_alpha = item_alpha
        self.user_alpha = user_alpha
        self.loss = loss
        self.trainer: Optional[BPRFMTrainer] = None

    def _create_trainer(self) -> BPRFMTrainer:
        return BPRFMTrainer(
            self.X_train_all,
            self.n_components,
            self.item_alpha,
            self.user_alpha,
            self.loss,
            self.n_threads,
        )

    def get_score(self, index: UserIndexArray) -> np.ndarray:
        if self.trainer is None:
            raise RuntimeError("get_score called before training")
        return (
            self.trainer.fm.user_embeddings[index].dot(
                self.trainer.fm.item_embeddings.T
            )
            + self.trainer.fm.item_biases[np.newaxis, :]
        )

    def get_score_block(self, begin: int, end: int) -> np.ndarray:
        if self.trainer is None:
            raise RuntimeError("get_score called before training")
        return (
            self.trainer.fm.user_embeddings[begin:end].dot(
                self.trainer.fm.item_embeddings.T
            )
            + self.trainer.fm.item_biases[np.newaxis, :]
        )

    def get_user_embedding(self) -> DenseMatrix:
        if self.trainer is None:
            raise RuntimeError("'get_user_embedding' called before training")
        return self.trainer.fm.user_embeddings.astype(np.float64)

    def get_score_from_user_embedding(
        self, user_embedding: DenseMatrix
    ) -> DenseScoreArray:
        if self.trainer is None:
            raise RuntimeError("'get_score_from_user_embedding' called before training")
        return (
            user_embedding.dot(self.trainer.fm.item_embeddings.T)
            + self.trainer.fm.item_biases[np.newaxis, :]
        )

    def get_item_embedding(self) -> DenseMatrix:
        if self.trainer is None:
            raise RuntimeError("'get_item_embedding' called before training")
        return self.trainer.fm.item_embeddings.astype(np.float64)

    def get_score_from_item_embedding(
        self, user_indices: UserIndexArray, item_embedding: DenseMatrix
    ) -> DenseScoreArray:
        if self.trainer is None:
            raise RuntimeError("'get_score_from_item_embedding' called before training")
        # ignore bias
        return (
            self.trainer.fm.user_embeddings[user_indices]
            .dot(item_embedding.T)
            .astype(np.float64)
        )
