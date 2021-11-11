import enum
import pickle
from typing import IO, Optional, Tuple

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
from ._ials import LossType
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
        alpha0: float,
        reg: float,
        nu: float,
        init_std: float,
        use_cg: bool,
        max_cg_steps: int,
        loss_type: LossType,
        random_seed: int,
        n_threads: int,
    ):
        X_train_all_f32 = X.astype(np.float32)
        config = (
            IALSLearningConfigBuilder()
            .set_K(n_components)
            .set_init_stdev(init_std)
            .set_alpha0(alpha0)
            .set_reg(reg)
            .set_nu(nu)
            .set_n_threads(n_threads)
            .set_use_cg(use_cg)
            .set_max_cg_steps(max_cg_steps)
            .set_loss_type(loss_type)
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


class IALSppConfig(BaseEarlyStoppingRecommenderConfig):
    n_components: int = 20
    alpha0: float = 1e-1
    scaled_reg: float = 1e-3
    nu: float = 1e-3
    init_std: float = 0.1
    use_cg: bool = True
    max_cg_steps: int = 3
    random_seed: int = 42
    n_threads: Optional[int] = None


class IALSConfig(BaseEarlyStoppingRecommenderConfig):
    n_components: int = 20
    alpha0: float = 1e-1
    scaled_reg: float = 1e-3
    nu: float = 1e-3
    init_std: float = 0.1
    use_cg: bool = True
    max_cg_steps: int = 3
    random_seed: int = 42
    n_threads: Optional[int] = None


def compute_reg_scale(X: sps.csr_matrix, alpha0: float, nu: float) -> float:
    X_csr: sps.csr_matrix = X.tocsr()
    X_csr.sort_indices()
    X_csc = X_csr.tocsc()
    X_csc.sort_indices()
    U, I = X.shape
    nnz_row: np.ndarray = X_csr.indptr[1:] - X_csr.indptr[:-1]
    nnz_col: np.ndarray = X_csc.indptr[1:] - X_csc.indptr[:-1]
    return float(((nnz_row + alpha0 * I) ** nu).sum()) + float(
        ((nnz_col + alpha0 * U) ** nu).sum()
    )


def ials_grad(
    X: sps.csr_matrix,
    u: np.ndarray,
    v: np.ndarray,
    reg: float,
    alpha: float,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    nu = u.shape[0]
    ni = v.shape[0]

    uv = u.dot(v.T)
    result_u = np.zeros_like(u)
    result_v = np.zeros_like(v)
    for uind in range(nu):
        for iind in range(ni):
            x = X[uind, iind]
            if x == 0:
                sc = uv[uind, iind]
            else:
                sc = (1 + alpha * np.log(1 + x / epsilon)) * (uv[uind, iind] - 1)

            result_u[uind, :] += v[iind] * sc
            result_v[iind, :] += u[uind] * sc
    result_u += reg * u
    result_v += reg * v
    return result_u, result_v


class IALSppRecommender(
    BaseRecommenderWithEarlyStopping,
    BaseRecommenderWithUserEmbedding,
    BaseRecommenderWithItemEmbedding,
):
    r"""Implementation of Implicit Alternating Least Squares(IALS) or Weighted Matrix Factorization(WMF).

    It tries to minimize the following loss:


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
        alpha0 (float, optional):
            The "unovserved" weight
        reg (float, optional) :
            Regularization coefficient for both user & item factors. Defaults to 1e-3.
        nu (float, optional) :
            Controlles frequency regularization.
        init_std (float, optional):
            Standard deviation for initialization normal distribution. Defaults to 0.1.
        use_cg (bool, optional):
            Whether to use the conjugate gradient method. Defaults to True.
        max_cg_steps (int, optional):
            Maximal number of conjute gradient descent steps. Defaults to 3.
            Ignored when ``use_cg=False``. By increasing this parameter, the result will be closer to
            Cholesky decomposition method (i.e., when ``use_cg = False``), but it wll take longer time.
        loss_type (irspack.recommenders._ials.LossType, optional):
            Specifies the subtle difference between iALS++ vs Original Loss.
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

    config_class = IALSppConfig
    nu_star = 1.0

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
        alpha0: float = 0.0,
        scaled_reg: float = 1e-3,
        nu: float = 1.0,
        confidence_scaling: str = "none",
        epsilon: float = 1.0,
        init_std: float = 0.1,
        use_cg: bool = True,
        max_cg_steps: int = 3,
        loss_type: LossType = LossType.IALSPP,
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
        self.alpha0 = alpha0
        self.scaled_reg = scaled_reg
        self.reg = (
            scaled_reg
            * compute_reg_scale(self.X_train_all, alpha0, self.nu_star)
            / compute_reg_scale(self.X_train_all, alpha0, nu)
        )
        print(f"self.reg is {self.reg}")
        self.nu = nu
        self.confidence_scaling = IALSConfigScaling[confidence_scaling]
        self.epsilon = epsilon

        self.init_std = init_std
        self.use_cg = use_cg
        self.max_cg_steps = max_cg_steps
        self.random_seed = random_seed
        self.n_threads = get_n_threads(n_threads)
        self.loss_type = loss_type

        self.trainer: Optional[IALSTrainer] = None

    def _create_trainer(self) -> TrainerBase:
        return IALSTrainer(
            self._scale_X(self.X_train_all, self.confidence_scaling, self.epsilon),
            self.n_components,
            self.alpha0,
            self.reg,
            self.nu,
            self.init_std,
            self.use_cg,
            self.max_cg_steps,
            self.loss_type,
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


class IALSRecommender(IALSppRecommender):
    r"""Implementation of Implicit Alternating Least Squares(IALS) or Weighted Matrix Factorization(WMF).

    It tries to minimize the following loss:


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
        alpha0 (float, optional):
            The "unovserved" weight
        reg (float, optional) :
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
            and if the variable is not set, it will be set to ``os.cpu_count()``. Defaults to None.
        max_epoch (int, optional):
            Maximal number of epochs. Defaults to 512.
    """

    config_class = IALSConfig
    nu_star = 0.0

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
            n_components=n_components,
            alpha0=1 / alpha,
            scaled_reg=reg / alpha,
            nu=0.0,
            init_std=init_std * (float(n_components) ** 0.5),
            confidence_scaling=confidence_scaling,
            epsilon=epsilon,
            use_cg=use_cg,
            max_cg_steps=max_cg_steps,
            loss_type=LossType.Original,
            random_seed=random_seed,
            validate_epoch=validate_epoch,
            score_degradation_max=score_degradation_max,
            n_threads=n_threads,
            max_epoch=max_epoch,
        )
