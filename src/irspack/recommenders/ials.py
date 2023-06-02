import enum
import logging
import pickle
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sps
from optuna import Study
from typing_extensions import Literal  # pragma: no cover, type: ignore

from .. import evaluation
from .._threading import get_n_threads
from ..definitions import (
    DenseMatrix,
    DenseScoreArray,
    InteractionMatrix,
    UserIndexArray,
)
from ..evaluation.evaluator import Evaluator
from ..optimization.parameter_range import (
    LogUniformFloatRange,
    ParameterRange,
    UniformIntegerRange,
)
from ._ials import IALSModelConfigBuilder, IALSSolverConfigBuilder
from ._ials import IALSTrainer as CoreTrainer
from ._ials import LossType, SolverType
from .base import (
    BaseRecommenderWithItemEmbedding,
    BaseRecommenderWithUserEmbedding,
    ParameterSuggestFunctionType,
)
from .base_earlystop import (
    BaseEarlyStoppingRecommenderConfig,
    BaseRecommenderWithEarlyStopping,
    TrainerBase,
)

if TYPE_CHECKING:
    from optuna import Study, Trial
    from optuna.storages import RDBStorage


def str_to_solver_type(t: str) -> SolverType:
    result: SolverType = getattr(SolverType, t.upper())
    assert result in {SolverType.CG, SolverType.CHOLESKY, SolverType.IALSPP}
    return result


def str_to_loss_type(t: str) -> LossType:
    result: LossType = getattr(LossType, t.upper())
    assert result in {LossType.ORIGINAL, LossType.IALSPP}
    return result


class IALSTrainer(TrainerBase):
    def __init__(
        self,
        X: InteractionMatrix,
        n_components: int,
        alpha0: float,
        reg: float,
        nu: float,
        init_std: float,
        solver_type: SolverType,
        max_cg_steps: int,
        ialspp_subspace_dimension: int,
        loss_type: LossType,
        random_seed: int,
        n_threads: int,
        prediction_time_max_cg_steps: int,
        prediction_time_ialspp_iteration: int,
    ):
        X_train_all_f32 = X.astype(np.float32)
        config = (
            IALSModelConfigBuilder()
            .set_K(n_components)
            .set_init_stdev(init_std)
            .set_alpha0(alpha0)
            .set_reg(reg)
            .set_nu(nu)
            .set_loss_type(loss_type)
            .set_random_seed(random_seed)
            .build()
        )
        solver_config = (
            IALSSolverConfigBuilder()
            .set_n_threads(n_threads)
            .set_solver_type(solver_type)
            .set_max_cg_steps(max_cg_steps)
            .set_ialspp_iteration(1)
            .set_ialspp_subspace_dimension(ialspp_subspace_dimension)
            .build()
        )
        self.core_trainer = CoreTrainer(config, X_train_all_f32)
        self.solver_config = solver_config
        self.prediction_time_solver_config = (
            IALSSolverConfigBuilder()
            .set_n_threads(n_threads)
            .set_solver_type(solver_type)
            .set_max_cg_steps(prediction_time_max_cg_steps)
            .set_ialspp_subspace_dimension(ialspp_subspace_dimension)
            .set_ialspp_iteration(prediction_time_ialspp_iteration)
            .build()
        )

    def load_state(self, ifs: IO) -> None:
        params = pickle.load(ifs)
        self.core_trainer.user = params["user"]
        self.core_trainer.item = params["item"]

    def compute_loss(self) -> float:
        return self.core_trainer.compute_loss(self.solver_config)

    def save_state(self, ofs: IO) -> None:
        pickle.dump(
            dict(user=self.core_trainer.user, item=self.core_trainer.item),
            ofs,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    def run_epoch(self) -> None:
        self.core_trainer.step(self.solver_config)

    def user_scores(self, begin: int, end: int) -> DenseScoreArray:
        return self.core_trainer.user_scores(begin, end, self.solver_config)

    def transform_user(self, X: InteractionMatrix) -> DenseMatrix:
        return self.core_trainer.transform_user(X, self.prediction_time_solver_config)

    def transform_item(self, X: InteractionMatrix) -> DenseMatrix:
        return self.core_trainer.transform_item(X, self.prediction_time_solver_config)


class IALSConfigScaling(enum.Enum):
    none = enum.auto()
    log = enum.auto()


class IALSConfig(BaseEarlyStoppingRecommenderConfig):
    n_components: int = 20
    alpha0: float = 1.0
    reg: float = 1e-3
    nu: float = 1.0
    confidence_scaling: str = "none"
    epsilon: float = 1.0
    init_std: float = 0.1
    solver_type: Literal["CG", "CHOLESKY"] = "CG"
    max_cg_steps: int = 3
    loss_type: Literal["IALSPP", "ORIGINAL"] = "IALSPP"
    nu_star: Optional[float] = None
    random_seed: int = 42
    n_threads: Optional[int] = None
    train_epochs: int = 16
    prediction_time_max_cg_steps: int = 5


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


class IALSRecommender(
    BaseRecommenderWithEarlyStopping,
    BaseRecommenderWithUserEmbedding,
    BaseRecommenderWithItemEmbedding,
):
    r"""Implementation of implicit Alternating Least Squares (iALS) or Weighted Matrix Factorization (WMF).

    By default, it tries to minimize the following loss:

    .. math ::

        \frac{1}{2} \sum _{u, i \in S}  c_{ui} (\mathbf{u}_u \cdot \mathbf{v}_i - 1) ^ 2
        + \frac{\alpha_0}{2} \sum_{u, i} (\mathbf{u}_u \cdot \mathbf{v}_i) ^ 2 +
        \frac{\text{reg}}{2} \left( \sum_u (\alpha_0 I + N_u) ^ \nu || \mathbf{u}_u || ^2 +                            \sum_i (\alpha_0 U + N_i) ^ \nu || \mathbf{v}_i || ^2 \right)

    where :math:`S` denotes the set of all pairs wher :math:`X_{ui}` is non-zero.

    See the seminal paper:

        - `Collaborative filtering for implicit feedback datasets
          <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.5120&rep=rep1&type=pdf>`_


    By default it uses a conjugate gradient descent version:

        - `Applications of the conjugate gradient method for implicit feedback collaborative filtering
          <https://dl.acm.org/doi/abs/10.1145/2043932.2043987>`_

    The loss above is slightly different from the original version. See the following paper for the loss used here

        - `Revisiting the Performance of iALS on Item Recommendation Benchmarks
          <https://arxiv.org/abs/2110.14037>`_

    Args:
        X_train_all (Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
            Input interaction matrix.
        n_components (int, optional):
            The dimension for latent factor. Defaults to 20.
        alpha0 (float, optional):
            The "unobserved" weight.
        reg (float, optional) :
            Regularization coefficient for both user & item factors. Defaults to 1e-3.
        nu (float, optional) :
            Controlles frequency regularization introduced in the paper,
            "Revisiting the Performance of iALS on Item Recommendation Benchmarks".
        confidence_scaling (str, optional) :
            Specifies how to scale confidence scaling :math:`c_{ui}`. Must be either "none" or "log".
            If "none", the non-zero (not-necessarily 1) :math:`X_{ui}` yields

            .. math ::
                c_{ui} = A + X_{ui}

            If "log",

            .. math ::
                c_{ui} = A + \log (1 + X_{ui} / \epsilon )

            The constant :math:`A` above will be 0 if ``loss_type`` is ``"IALSPP"``, :math:`\alpha_0` if ``loss_type`` is ``"ORIGINAL"``.

            Defaults to "none".
        epsilon (float, optional):
            The :math:`\epsilon` parameter for log-scaling described above.
            Will not have any effect if `confidence_scaling` is "none".
            Defaults to 1.0f.
        init_std (float, optional):
            Standard deviation for initialization normal distribution.
            The actual std for each user/item vector components are scaled by `1 / n_components ** .5`.
            Defaults to 0.1.
        solver_type ( "CHOLESKY" | "CG" | "IALSPP", optional):
            Which solver to use. Defaults to "CG".
        max_cg_steps (int, optional):
            Maximal number of conjute gradient descent steps during the training time.
            Defaults to 3.
            Used only when ``solver_type=="CG"``.
            By increasing this parameter, the result will be closer to
            Cholesky decomposition method (i.e., when ``solver_type == "CHOLESKY"``), but it wll take longer time.
        ialspp_subspace_dimension (int, optional):
            The subspace dimension of iALS++ (ignored if the ``solver_type`` is not "IALSPP").
            If this value is 1, specialized implementation described in `Fast Matrix Factorization for Online Recommendation with Implicit Feedback <https://arxiv.org/abs/1708.05024>`_ will be used instead.
            Defaults to 64.
        loss_type ( Literal["IALSPP", "ORIGINAL"], optional):
            Specifies the subtle difference between iALS++ vs Original Loss.
        nu_star (Optional[float], optional):
            If not `None`, used as the reference scale for nu described in the "Revisiting..." paper.
            Defaults to None.
        random_seed (int, optional):
            The random seed to initialize the parameters.
        n_threads (Optional[int], optional):
            Specifies the number of threads to use for the computation.
            If ``None``, the environment variable ``"IRSPACK_NUM_THREADS_DEFAULT"`` will be looked up,
            and if the variable is not set, it will be set to ``os.cpu_count()``. Defaults to None.
        train_epochs (int, optional):
            Maximal number of epochs. Defaults to 16.
        prediction_time_max_cg_steps (int, optional):
            Maximal number of conjute gradient descent steps during the prediction time,
            i.e., the case when a user unseen at the training time is given as a history matrix.
            Defaults to 5.

    Examples:

        >>> from irspack import IALSRecommender, rowwise_train_test_split, Evaluator
        >>> from irspack.utils.sample_data import mf_example_data
        >>> X = mf_example_data(100, 30, random_state=1)
        >>> X_train, X_test = rowwise_train_test_split(X, random_state=0)
        >>> rec = IALSRecommender(X_train)
        >>> rec.learn()
        >>> evaluator=Evaluator(X_test)
        >>> print(evaluator.get_scores(rec, [20]))
        OrderedDict([('hit@20', 1.0), ('recall@20', 0.9003412698412698), ('ndcg@20', 0.6175493479217139), ('map@20', 0.3848785870622406), ('precision@20', 0.3385), ('gini_index@20', 0.0814), ('entropy@20', 3.382497875272383), ('appeared_item@20', 30.0)])
    """

    config_class = IALSConfig
    default_tune_range: List[ParameterRange] = [
        UniformIntegerRange("n_components", 4, 300),
        LogUniformFloatRange("alpha0", 3e-3, 1),
        LogUniformFloatRange("reg", 1e-4, 1e-1),
    ]

    def __init__(
        self,
        X_train_all: InteractionMatrix,
        n_components: int = 20,
        alpha0: float = 0.0,
        reg: float = 1e-3,
        nu: float = 1.0,
        confidence_scaling: str = "none",
        epsilon: float = 1.0,
        init_std: float = 0.1,
        solver_type: Literal["CG", "CHOLESKY", "IALSPP"] = "CG",
        max_cg_steps: int = 3,
        ialspp_subspace_dimension: int = 64,
        loss_type: Literal["IALSPP", "ORIGINAL"] = "IALSPP",
        nu_star: Optional[float] = None,
        random_seed: int = 42,
        n_threads: Optional[int] = None,
        train_epochs: int = 16,
        prediction_time_max_cg_steps: int = 5,
        prediction_time_ialspp_iteration: int = 7,
    ) -> None:

        super().__init__(
            X_train_all,
            train_epochs=train_epochs,
        )

        self.n_components = n_components
        self.alpha0 = alpha0
        self.reg = reg

        self.nu = nu
        self.confidence_scaling = IALSConfigScaling[confidence_scaling]
        self.epsilon = epsilon

        self.init_std = init_std
        self.solver_type = str_to_solver_type(solver_type)
        self.max_cg_steps = max_cg_steps
        self.ialspp_subspace_dimension = ialspp_subspace_dimension
        self.random_seed = random_seed
        self.n_threads = get_n_threads(n_threads)
        self.loss_type = str_to_loss_type(loss_type)
        self.nu_star = nu_star

        self.scaled_reg = self.reg
        if self.nu_star is not None:
            self.scaled_reg = (
                self.reg
                * compute_reg_scale(self.X_train_all, alpha0, self.nu_star)
                / compute_reg_scale(self.X_train_all, alpha0, nu)
            )
        self.prediction_time_max_cg_steps = prediction_time_max_cg_steps
        self.prediction_time_ialspp_iteration = prediction_time_ialspp_iteration

        self.trainer: Optional[IALSTrainer] = None

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

    def _create_trainer(self) -> TrainerBase:
        return IALSTrainer(
            X=self._scale_X(self.X_train_all, self.confidence_scaling, self.epsilon),
            n_components=self.n_components,
            alpha0=self.alpha0,
            reg=self.scaled_reg,
            nu=self.nu,
            init_std=self.init_std,
            solver_type=self.solver_type,
            max_cg_steps=self.max_cg_steps,
            ialspp_subspace_dimension=self.ialspp_subspace_dimension,
            loss_type=self.loss_type,
            random_seed=self.random_seed,
            n_threads=self.n_threads,
            prediction_time_max_cg_steps=self.prediction_time_max_cg_steps,
            prediction_time_ialspp_iteration=self.prediction_time_ialspp_iteration,
        )

    @property
    def trainer_as_ials(self) -> IALSTrainer:
        if self.trainer is None:
            raise RuntimeError("tried to fetch trainer before the training.")
        return self.trainer

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        res: DenseScoreArray = self.trainer_as_ials.core_trainer.user[user_indices].dot(
            self.get_item_embedding().T
        )
        return res

    def get_score_block(self, begin: int, end: int) -> DenseScoreArray:
        return self.trainer_as_ials.user_scores(begin, end)

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        user_vector = self.compute_user_embedding(X)
        return self.get_score_from_user_embedding(user_vector)

    def get_user_embedding(self) -> DenseMatrix:
        return self.trainer_as_ials.core_trainer.user

    def get_score_from_user_embedding(
        self, user_embedding: DenseMatrix
    ) -> DenseScoreArray:
        res: DenseScoreArray = user_embedding.dot(self.get_item_embedding().T)
        return res

    def get_item_embedding(self) -> DenseMatrix:
        return self.trainer_as_ials.core_trainer.item

    def compute_user_embedding(self, X: InteractionMatrix) -> DenseMatrix:
        r"""Given an unknown users' interaction with known items,
        computes the latent factors of the users by least square (fixing item embeddings).

        parameters:
            X:
                The interaction history of the new users.
                ``X.shape[1]`` must be equal to ``self.n_items``.
        """
        return self.trainer_as_ials.transform_user(
            self._scale_X(
                sps.csr_matrix(X).astype(np.float32),
                self.confidence_scaling,
                self.epsilon,
            )
        )

    def compute_item_embedding(self, X: InteractionMatrix) -> DenseMatrix:
        r"""Given an unknown items' interaction with known user,
        computes the latent factors of the items by least square (fixing user embeddings).

        parameters:
            X:
                The interaction history of the new users.
                ``X.shape[0]`` must be equal to ``self.n_users``.
        """

        return self.trainer_as_ials.transform_item(
            self._scale_X(
                sps.csr_matrix(X).astype(np.float32),
                self.confidence_scaling,
                self.epsilon,
            )
        )

    def get_score_from_item_embedding(
        self, user_indices: UserIndexArray, item_embedding: DenseMatrix
    ) -> DenseScoreArray:
        res: DenseScoreArray = self.trainer_as_ials.core_trainer.user[user_indices].dot(
            item_embedding.T
        )
        return res

    @classmethod
    def tune_doubling_dimension(
        cls,
        data: InteractionMatrix,
        evaluator: Evaluator,
        initial_dimension: int,
        maximal_dimension: int,
        storage: Optional["RDBStorage"] = None,
        study_name_prefix: Optional[str] = None,
        n_trials_initial: int = 40,
        n_trials_following: int = 20,
        n_startup_trials_initial: int = 10,
        n_startup_trials_following: int = 5,
        max_epoch: int = 16,
        validate_epoch: int = 1,
        score_degradation_max: int = 3,
        neighborhood_scale: float = 3.0,
        suggest_function_initial: Optional[ParameterSuggestFunctionType] = None,
        random_seed: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        r"""Perform tuning gradually doubling `n_components`.
        Typically, with the initial `n_components`, the search will be more exhaustive, and with larger `n_components`, less exploration will be done around previously found parameters.
        This strategy is described in `Revisiting the Performance of iALS on Item Recommendation Benchmarks <https://arxiv.org/abs/2110.14037>`_.

        Args:
            initial_dimension: The initial dimension.
            maximal_dimension: The maximal (inclusive) dimension to be tried.
            storage:
                The storage where multiple `optuna.Study` will be created corresponding to the various dimensions.
                If `None`, all `Study` will be created in-memory.
            study_name_prefix:
                The prefix for the names of `optuna.Study`. For dimension `d`, the full name of the `Study` will be `"{study_name_prefix}_{d}"`.
                If `None`, we will use a random string for this prefix.
            n_trials_initial:
                The number of trials for the initial dimension.
            n_trials_following:
                The number of trials for the following dimensions.
            n_startup_trials_initial:
                Passed on to `n_startup_trials` argument of `optuna.pruners.MedianPruner` in the initial `optuna.Study`.
                Defaults to `10`.
            n_startup_trials_following:
                Passed on to `n_startup_trials` argument of `optuna.pruners.MedianPruner` in the following `optuna.Study`.
                Defaults to `5`.
            neighborhood_scale:
                `alpha_0` and `reg` parameters will be searched within the log-uniform range
                [`previous_dimension_result / neighborhood_scale`, `previous_dimension_result * neighborhood_scale`].
                Defaults to `3.0`
            suggest_overwrite_initial:
                Overwrites the suggestion parameters in the initial `optuna.Study`.
                Defaults to `[]`.
            random_seed:
                The random seed to control ``optuna.samplers.TPESampler``. Defaults to `None`.

        Returns:
            A tuple that consists of
                1. A dict containing the best paramaters.
                   This dict can be passed to the recommender as ``**kwargs``.
                2. A ``pandas.DataFrame`` that contains the history of optimization for all dimensions.
        """
        from uuid import uuid1

        import optuna

        if study_name_prefix is None:
            study_name_prefix = str(uuid1())
        dimension = initial_dimension
        prev_params: Optional[Dict[str, Any]] = None
        results: List[Tuple[float, Dict[str, Any], pd.DataFrame]] = []
        while dimension <= maximal_dimension:
            _suggest: Optional[ParameterSuggestFunctionType]
            if random_seed is not None:
                random_seed += 1
            if prev_params is None:
                _suggest = suggest_function_initial
                n_trials = n_trials_initial
                n_startup_trials = n_startup_trials_initial
            else:
                n_trials = n_trials_following
                n_startup_trials = n_startup_trials_following

                _prev_params_copied = prev_params.copy()

                def _suggest(trial: "Trial") -> Dict[str, Any]:
                    result: Dict[str, Any] = {}
                    for key, value in _prev_params_copied.items():
                        if isinstance(value, float):
                            result[key] = trial.suggest_float(
                                key,
                                value / neighborhood_scale,
                                value * neighborhood_scale,
                                log=True,
                            )
                    return result

            study = optuna.create_study(
                storage=storage,
                sampler=optuna.samplers.TPESampler(seed=random_seed),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials),
                study_name=f"{study_name_prefix}_{dimension}",
            )
            bp, df_ = cls.tune_with_study(
                study,
                data,
                evaluator,
                n_trials,
                max_epoch=max_epoch,
                validate_epoch=validate_epoch,
                score_degradation_max=score_degradation_max,
                parameter_suggest_function=_suggest,
                fixed_params=dict(n_components=dimension),
            )
            df_["n_components"] = dimension
            results.append((float(df_["value"].min()), bp, df_))
            prev_params = study.best_params.copy()
            dimension *= 2
        final_result_df = pd.concat([x[2] for x in results])
        final_bp = sorted(results)[0][1]

        return final_bp, final_result_df

    @classmethod
    def tune_with_study(
        cls,
        study: "Study",
        data: Union[InteractionMatrix, None],
        evaluator: "evaluation.Evaluator",
        n_trials: int = 20,
        timeout: Optional[int] = None,
        data_suggest_function: Optional[Callable[["Trial"], InteractionMatrix]] = None,
        parameter_suggest_function: Optional[ParameterSuggestFunctionType] = None,
        fixed_params: Dict[str, Any] = dict(),
        max_epoch: int = 16,
        validate_epoch: int = 1,
        score_degradation_max: int = 3,
        logger: Optional[logging.Logger] = None,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        return super().tune_with_study(
            study,
            data,
            evaluator,
            n_trials,
            timeout,
            data_suggest_function,
            parameter_suggest_function,
            fixed_params,
            max_epoch,
            validate_epoch,
            score_degradation_max,
            logger,
        )

    @classmethod
    def tune(
        cls,
        data: Union[InteractionMatrix, None],
        evaluator: "evaluation.Evaluator",
        n_trials: int = 20,
        timeout: Optional[int] = None,
        data_suggest_function: Optional[Callable[["Trial"], InteractionMatrix]] = None,
        parameter_suggest_function: Optional[ParameterSuggestFunctionType] = None,
        fixed_params: Dict[str, Any] = dict(),
        random_seed: Optional[int] = None,
        prunning_n_startup_trials: int = 10,
        max_epoch: int = 16,
        validate_epoch: int = 1,
        score_degradation_max: int = 3,
        logger: Optional[logging.Logger] = None,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        return super().tune(
            data,
            evaluator,
            n_trials,
            timeout,
            data_suggest_function,
            parameter_suggest_function,
            fixed_params,
            random_seed,
            prunning_n_startup_trials,
            max_epoch,
            validate_epoch,
            score_degradation_max,
            logger,
        )
