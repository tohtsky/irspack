import enum
import pickle
from typing import IO, Optional

import numpy as np
import scipy.sparse as sps

from irspack.utils import get_n_threads

from ..definitions import (
    DenseMatrix,
    DenseScoreArray,
    InteractionMatrix,
    UserIndexArray,
)
from ._mf import NMFLearningConfigBuilder
from ._mf import NMFTrainer as CoreTrainer
from .base import BaseRecommenderWithItemEmbedding, BaseRecommenderWithUserEmbedding
from .base_earlystop import (
    BaseEarlyStoppingRecommenderConfig,
    BaseRecommenderWithEarlyStopping,
    TrainerBase,
)


class NMFTrainer(TrainerBase):
    def __init__(
        self,
        X: InteractionMatrix,
        n_components: int,
        l2_reg: float,
        l1_reg: float,
        shuffle: bool,
        random_seed: int,
        n_threads: int,
    ):
        X_train_all_f32 = X.astype(np.float32)
        config = (
            NMFLearningConfigBuilder()
            .set_K(n_components)
            .set_l2_reg(l2_reg)
            .set_l1_reg(l1_reg)
            .set_n_threads(n_threads)
            .set_shuffle(shuffle)
            .set_random_seed(random_seed)
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


class NMFConfig(BaseEarlyStoppingRecommenderConfig):
    n_components: int = 20
    l2_reg: float = 0.0
    l1_reg: float = 0.0
    shuffle: bool = True
    random_seed: int = 42
    n_threads: Optional[int] = None


class NMFRecommender(
    BaseRecommenderWithEarlyStopping,
    BaseRecommenderWithUserEmbedding,
    BaseRecommenderWithItemEmbedding,
):
    r"""Implementation of non-negative matrix factorization (NMF).

    It tries to minimize the following loss:

    .. math ::

        \frac{1}{2} \sum _{u, i}  (\mathbf{w}_u \cdot \mathbf{h}_i - X_{ui}) ^ 2 +
        \frac{\text{alpha}(1 - \text{l1\_ratio})}{2} \left( \sum _u || \mathbf{w}_u || ^2 + \sum _i || \mathbf{h}_i || ^2 \right) +
        \text{alpha}(\text{l1\_ratio}) \left( \sum _u | \mathbf{u}_u | + \sum _i | \mathbf{h}_i | \right)


    Args:
        X_train_all (Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
            Input interaction matrix.
        n_components (int, optional):
            The dimension for latent factor. Defaults to 20.
        alpha (float, optional):
            Controlls overall regularization magnitude. Defaults to 0.0.
        l1_ratio (float, optional) :
            The ratio of L1 regularization coefficient relative to `alpha`. Defaults to 0.
        shuffle (bool, optional):
            Whether to shuffle the coordinate descent ordering. Defaults to True.
        validate_epoch (int, optional):
            Frequency of validation score measurement (if any). Defaults to 5.
        score_degradation_max (int, optional):
            Maximal number of allowed score degradation. Defaults to 5.
        n_threads (Optional[int], optional):
            Specifies the number of threads to use for the computation.
            If ``None``, the environment variable ``"IRSPACK_NUM_THREADS_DEFAULT"`` will be looked up,
            and if the variable is not set, it will be set to ``os.cpu_count()``. Defaults to None.
        max_epoch (int, optional):
            Maximal number of epochs. Defaults to 512.
    """

    config_class = NMFConfig

    def __init__(
        self,
        X_train_all: InteractionMatrix,
        n_components: int = 20,
        alpha: float = 0.0,
        l1_ratio: float = 0,
        shuffle: bool = True,
        random_seed: int = 42,
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
        self.l1_ratio = l1_ratio
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.n_threads = get_n_threads(n_threads)

        self.trainer: Optional[NMFTrainer] = None

    def _create_trainer(self) -> TrainerBase:
        return NMFTrainer(
            self.X_train_all,
            self.n_components,
            self.alpha * (1 - self.l1_ratio),
            self.alpha * self.l1_ratio,
            self.shuffle,
            self.random_seed,
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
        return self.core_trainer.user

    def get_score_from_user_embedding(
        self, user_embedding: DenseMatrix
    ) -> DenseScoreArray:
        return user_embedding.dot(self.get_item_embedding().T)

    def get_item_embedding(self) -> DenseMatrix:
        return self.core_trainer.item

    def compute_user_embedding(self, X: InteractionMatrix) -> DenseMatrix:
        r"""Given an unknown users' interaction with known items,
        computes the latent factors of the users by least square (fixing item embeddings).

        Parameters:
            X:
                The interaction history of the new users.
                ``X.shape[1]`` must be equal to ``self.n_items``.
        """
        return self.core_trainer.transform_user(X)

    def compute_item_embedding(self, X: InteractionMatrix) -> DenseMatrix:
        r"""Given an unknown items' interaction with known user,
        computes the latent factors of the items by least square (fixing user embeddings).

        Parameters:
            X:
                The interaction history of the new users.
                ``X.shape[0]`` must be equal to ``self.n_users``.
        """

        return self.core_trainer.transform_item(X)

    def get_score_from_item_embedding(
        self, user_indices: UserIndexArray, item_embedding: DenseMatrix
    ) -> DenseScoreArray:
        return self.core_trainer.user[user_indices].dot(item_embedding.T)
