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
from ._ials import IALSLearningConfigBuilder
from ._ials import IALSTrainer as CoreTrainer
from .base import BaseRecommenderWithItemEmbedding, BaseRecommenderWithUserEmbedding
from .base_earlystop import (
    BaseEarlyStoppingRecommenderConfig,
    BaseRecommenderWithEarlyStopping,
    TrainerBase,
)


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
        random_seed: int,
        n_threads: int,
    ):
        X_train_all_f32 = X.astype(np.float32)
        config = (
            IALSLearningConfigBuilder()
            .set_K(n_components)
            .set_init_stdev(init_std)
            .set_alpha(alpha)
            .set_reg(reg)
            .set_n_threads(n_threads)
            .set_use_cg(use_cg)
            .set_max_cg_steps(max_cg_steps)
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


class IALSConfigScaling(enum.Enum):
    none = enum.auto()
    log = enum.auto()


class IALSConfig(BaseEarlyStoppingRecommenderConfig):
    n_components: int = 20
    alpha: float = 0.0
    reg: float = 1e-3
    confidence_scaling: str = "none"
    epsilon: float = 1.0
    init_std: float = 0.1
    use_cg: bool = True
    max_cg_steps: int = 3
    random_seed: int = 42
    n_threads: Optional[int] = None


class IALSRecommender(
    BaseRecommenderWithEarlyStopping,
    BaseRecommenderWithUserEmbedding,
    BaseRecommenderWithItemEmbedding,
):
    r"""Implementation of Implicit Alternating Least Squares(IALS) or Weighted Matrix Factorization(WMF).

    It tries to minimize the following loss:

    .. math ::

        \frac{1}{2} \sum _{u, i} c_{ui} (\mathbf{u}_u \cdot \mathbf{v}_i - \mathbb{1}_{r_{ui} > 0}) ^ 2 +
        \frac{\text{reg}}{2} \left( \sum _u || \mathbf{u}_u || ^2 + \sum _i || \mathbf{v}_i || ^2 \right)


    See the seminal paper:

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
        reg (float, optional) :
            Regularization coefficient for both user & item factors. Defaults to 1e-3.
        confidence_scaling (str, optional) :
            Specifies how to scale confidence scaling :math:`c_{ui}`. Must be either "none" or "log".
            If "none", the non-zero "rating" :math:`r_{ui}` yields

            .. math ::

                c_{ui} = 1 + \alpha r_{ui}

            If "log",

            .. math ::

                c_{ui} = 1 + \alpha \log (1 + r_{ui} / \epsilon )

            Defaults to "none".
        epsilon (float, optional):
            The :math:`\epsilon` parameter for log-scaling described above.
            Will not have any effect if `confidence_scaling` is "none".
            Defaults to 1.0f.
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
            and if the variable is not set, it will be set to ``os.cpu_count()``. Defaults to None.
        max_epoch (int, optional):
            Maximal number of epochs. Defaults to 512.
    """

    config_class = IALSConfig

    @classmethod
    def _scale_X(
        cls, X: sps.csr_matrix, scheme: IALSConfigScaling, epsilon: float
    ) -> sps.csr_matrix:
        if scheme is IALSConfigScaling.none:
            return X
        else:
            X_ret: sps.csr_matrix = X.copy()
            X_ret.data = np.log(1 + X_ret.data / epsilon)
            return X_ret

    def __init__(
        self,
        X_train_all: InteractionMatrix,
        n_components: int = 20,
        alpha: float = 0.0,
        reg: float = 1e-3,
        confidence_scaling: str = "none",
        epsilon: float = 1.0,
        init_std: float = 0.1,
        use_cg: bool = True,
        max_cg_steps: int = 3,
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
        self.reg = reg
        self.init_std = init_std
        self.use_cg = use_cg
        self.max_cg_steps = max_cg_steps
        self.confidence_scaling = IALSConfigScaling[confidence_scaling]
        self.epsilon = epsilon
        self.random_seed = random_seed
        self.n_threads = get_n_threads(n_threads)

        self.trainer: Optional[IALSTrainer] = None

    def _create_trainer(self) -> TrainerBase:
        return IALSTrainer(
            self._scale_X(self.X_train_all, self.confidence_scaling, self.epsilon),
            self.n_components,
            self.alpha,
            self.reg,
            self.init_std,
            self.use_cg,
            self.max_cg_steps,
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
        return self.core_trainer.transform_user(
            self._scale_X(
                sps.csr_matrix(X).astype(np.float32),
                self.confidence_scaling,
                self.epsilon,
            )
        )

    def compute_item_embedding(self, X: InteractionMatrix) -> DenseMatrix:
        r"""Given an unknown items' interaction with known user,
        computes the latent factors of the items by least square (fixing user embeddings).

        Parameters:
            X:
                The interaction history of the new users.
                ``X.shape[0]`` must be equal to ``self.n_users``.
        """

        return self.core_trainer.transform_item(
            self._scale_X(
                sps.csr_matrix(X).astype(np.float32),
                self.confidence_scaling,
                self.epsilon,
            )
        )

    def get_score_from_item_embedding(
        self, user_indices: UserIndexArray, item_embedding: DenseMatrix
    ) -> DenseScoreArray:
        return self.core_trainer.user[user_indices].dot(item_embedding.T)
