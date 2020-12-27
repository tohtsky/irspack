import pickle
from typing import IO, Optional

import numpy as np

from ..definitions import (
    DenseMatrix,
    DenseScoreArray,
    InteractionMatrix,
    UserIndexArray,
)
from ._ials import IALSLearningConfigBuilder
from ._ials import IALSTrainer as CoreTrainer
from .base import (
    BaseRecommenderWithItemEmbedding,
    BaseRecommenderWithThreadingSupport,
    BaseRecommenderWithUserEmbedding,
)
from .base_earlystop import BaseRecommenderWithEarlyStopping, TrainerBase


class IALSTrainer(TrainerBase):
    def __init__(
        self,
        X: InteractionMatrix,
        n_components: int,
        alpha: float,
        reg: float,
        init_std: float,
        use_cg: bool,
        max_cg_steps: int,
        n_thread: int,
    ):
        X_all_f32 = X.astype(np.int32)
        config = (
            IALSLearningConfigBuilder()
            .set_K(n_components)
            .set_init_stdev(init_std)
            .set_alpha(alpha)
            .set_reg(reg)
            .set_n_threads(n_thread)
            .set_use_cg(use_cg)
            .set_max_cg_steps(max_cg_steps)
            .build()
        )

        self.core_trainer = CoreTrainer(config, X_all_f32)

    def load_state(self, ifs: IO) -> None:
        params = pickle.load(ifs)
        self.core_trainer.user = params["user"]
        self.core_trainer.item = params["item"]

    def save_state(self, ofs: IO) -> None:
        pickle.dump(
            dict(user=self.core_trainer.user, item=self.core_trainer.item),
            ofs,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    def run_epoch(self) -> None:
        self.core_trainer.step()


class IALSRecommender(
    BaseRecommenderWithEarlyStopping,
    BaseRecommenderWithThreadingSupport,
    BaseRecommenderWithUserEmbedding,
    BaseRecommenderWithItemEmbedding,
):
    """
    Implicit Alternating Least Squares (IALS).
    See:
    Y. Hu, Y. Koren and C. Volinsky, Collaborative filtering for implicit feedback datasets, ICDM 2008.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.5120&rep=rep1&type=pdf
    """

    def __init__(
        self,
        X_all: InteractionMatrix,
        n_components: int = 20,
        alpha: float = 1.0,
        reg: float = 1e-3,
        init_std: float = 0.1,
        use_cg: bool = True,
        max_cg_steps: int = 3,
        validate_epoch: int = 5,
        score_degradation_max: int = 5,
        n_thread: Optional[int] = 1,
        max_epoch: int = 300,
    ):
        super().__init__(
            X_all,
            max_epoch=max_epoch,
            validate_epoch=validate_epoch,
            score_degration_max=score_degradation_max,
            n_thread=n_thread,
        )

        self.n_components = n_components
        self.alpha = alpha
        self.reg = reg
        self.init_std = init_std
        self.use_cg = use_cg
        self.max_cg_steps = max_cg_steps

        self.trainer: Optional[IALSTrainer] = None

    def create_trainer(self) -> TrainerBase:
        return IALSTrainer(
            self.X_all,
            self.n_components,
            self.alpha,
            self.reg,
            self.init_std,
            self.use_cg,
            self.max_cg_steps,
            self.n_thread,
        )

    def get_score(self, index: UserIndexArray) -> DenseScoreArray:
        if self.trainer is None:
            raise RuntimeError("'get_score' called before training")
        return self.trainer.core_trainer.user[index].dot(
            self.trainer.core_trainer.item.T
        )

    def get_score_block(self, begin: int, end: int) -> DenseScoreArray:
        if self.trainer is None:
            raise RuntimeError("'get_score_block' called before training")
        return self.trainer.core_trainer.user_scores(begin, end)

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        if self.trainer is None:
            raise RuntimeError("'get_score_cols_user' called before training")
        user_vector = self.trainer.core_trainer.transform_user(
            X.astype(np.float32).tocsr()
        )
        return user_vector.dot(self.trainer.core_trainer.item.T).astype(np.float64)

    def get_user_embedding(self) -> DenseMatrix:
        if self.trainer is None:
            raise RuntimeError("'get_user_embedding' called before training")

        return self.trainer.core_trainer.user.astype(np.float64)

    def get_score_from_user_embedding(
        self, user_embedding: DenseMatrix
    ) -> DenseScoreArray:
        if self.trainer is None:
            raise RuntimeError("'get_score_from_user_embedding' called before training")

        return user_embedding.dot(self.trainer.core_trainer.item.T)

    def get_item_embedding(self) -> DenseMatrix:
        if self.trainer is None:
            raise RuntimeError("'get_item_embedding' called before training")
        return self.trainer.core_trainer.item.astype(np.float64)

    def get_score_from_item_embedding(
        self, user_indices: UserIndexArray, item_embedding: DenseMatrix
    ) -> DenseScoreArray:
        if self.trainer is None:
            raise RuntimeError("'get_score_from_item_embedding' called before training")
        return (
            self.trainer.core_trainer.user[user_indices]
            .dot(item_embedding.T)
            .astype(np.float64)
        )
