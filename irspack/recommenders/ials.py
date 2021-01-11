import pickle
from typing import IO, Optional

import numpy as np

from irspack.utils import get_n_threads

from ..definitions import (
    DenseMatrix,
    DenseScoreArray,
    InteractionMatrix,
    UserIndexArray,
)
from ._ials import IALSLearningConfigBuilder
from ._ials import IALSTrainer as CoreTrainer
from .base import BaseRecommenderWithItemEmbedding, BaseRecommenderWithUserEmbedding
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
        n_threads: int,
    ):
        X_train_all_f32 = X.astype(np.int32)
        config = (
            IALSLearningConfigBuilder()
            .set_K(n_components)
            .set_init_stdev(init_std)
            .set_alpha(alpha)
            .set_reg(reg)
            .set_n_threads(n_threads)
            .set_use_cg(use_cg)
            .set_max_cg_steps(max_cg_steps)
            .build()
        )

        self.core_trainer = CoreTrainer(config, X_train_all_f32)

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
    BaseRecommenderWithUserEmbedding,
    BaseRecommenderWithItemEmbedding,
):
    """Implementation of Implicit Alternating Least Squares(IALS) or Weighted Matrix Factorization(WMF).

    See:

        - `Collaborative filtering for implicit feedback datasets
          <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.5120&rep=rep1&type=pdf>`_

    To speed up the learning procedure, we have also implemented the conjugate gradient descent version following:

        - `Applications of the conjugate gradient method for implicit feedback collaborative filtering
          <https://dl.acm.org/doi/abs/10.1145/2043932.2043987>`_


    Args:
        X_train_all (Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
            Input interaction matrix.
        n_components (int, optional):
            The dimension for latent factor. Defaults to 20.
        alpha (float, optional):
            The confidence parameter alpha in the original paper. Defaults to 0.0.
        reg (float, optional):
            Regularization coefficient for both user & item factors. Defaults to 1e-3.
        init_std (float, optional):
            Standard deviation for initialization normal distribution. Defaults to 0.1.
        use_cg (bool, optional):
            Whether to use the conjugate gradient method. Defaults to True.
        max_cg_steps (int, optional):
            Maximal number of conjute gradient descent steps. Defaults to 3.
            Ignored when ``use_cg=False``. By increasing this parameter, the result will be closer to
            Cholesky decomposition method (i.e., when ``use_cg = False``), but it wll take longer time.
        validate_epoch (int, optional):
            Frequency of validation score measurement (if any). Defaults to 5.
        score_degradation_max (int, optional):
            Maximal number of allowed score degradation. Defaults to 5.
        n_threads (Optional[int], optional):
            Specifies the number of threads to use for the computation.
            If ``None``, the environment variable ``"IRSPACK_NUM_THREADS_DEFAULT"`` will be looked up,
            and if there is no such an environment variable, it will be set to 1. Defaults to None.
        max_epoch (int, optional):
            Maximal number of epochs. Defaults to 512.
    """

    def __init__(
        self,
        X_train_all: InteractionMatrix,
        n_components: int = 20,
        alpha: float = 0.0,
        reg: float = 1e-3,
        init_std: float = 0.1,
        use_cg: bool = True,
        max_cg_steps: int = 3,
        validate_epoch: int = 5,
        score_degradation_max: int = 5,
        n_threads: Optional[int] = None,
        max_epoch: int = 512,
    ) -> None:

        super().__init__(
            X_train_all,
            max_epoch=max_epoch,
            validate_epoch=validate_epoch,
            score_degradation_max=score_degradation_max,
        )

        self.n_components = n_components
        self.alpha = alpha
        self.reg = reg
        self.init_std = init_std
        self.use_cg = use_cg
        self.max_cg_steps = max_cg_steps
        self.n_threads = get_n_threads(n_threads)

        self.trainer: Optional[IALSTrainer] = None

    def _create_trainer(self) -> TrainerBase:
        return IALSTrainer(
            self.X_train_all,
            self.n_components,
            self.alpha,
            self.reg,
            self.init_std,
            self.use_cg,
            self.max_cg_steps,
            self.n_threads,
        )

    @property
    def core_trainer(self) -> CoreTrainer:
        if self.trainer is None:
            raise RuntimeError("tried to fetch core_trainer before the training.")
        return self.trainer.core_trainer

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        return self.core_trainer.user[user_indices].dot(self.get_item_embedding().T)

    def get_score_block(self, begin: int, end: int) -> DenseScoreArray:
        return self.core_trainer.user_scores(begin, end)

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        user_vector = self.compute_user_embedding(X)
        return self.get_score_from_user_embedding(user_vector)

    def get_user_embedding(self) -> DenseMatrix:
        return self.core_trainer.user.astype(np.float64)

    def get_score_from_user_embedding(
        self, user_embedding: DenseMatrix
    ) -> DenseScoreArray:
        return user_embedding.dot(self.get_item_embedding().T).astype(np.float64)

    def get_item_embedding(self) -> DenseMatrix:
        return self.core_trainer.item.astype(np.float64)

    def compute_user_embedding(self, X: InteractionMatrix) -> DenseMatrix:
        return self.core_trainer.transform_user(X.astype(np.float32).tocsr())

    def compute_item_embedding(self, X: InteractionMatrix) -> DenseMatrix:
        return self.core_trainer.transform_item(X.astype(np.float32).tocsr())

    def get_score_from_item_embedding(
        self, user_indices: UserIndexArray, item_embedding: DenseMatrix
    ) -> DenseScoreArray:
        return (
            self.core_trainer.user[user_indices]
            .dot(item_embedding.T)
            .astype(np.float64)
        )
