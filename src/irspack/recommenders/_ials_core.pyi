"""
irspack's core module for "IALSRecommender".
Built to use
        ARM NEON
"""

import enum
from typing import Annotated, overload

import numpy
import scipy
from numpy.typing import NDArray

class LossType(enum.Enum):
    ORIGINAL = 0

    IALSPP = 1

class SolverType(enum.Enum):
    CHOLESKY = 0

    CG = 1

    IALSPP = 2

ORIGINAL: LossType = LossType.ORIGINAL

CHOLESKY: SolverType = SolverType.CHOLESKY

CG: SolverType = SolverType.CG

IALSPP: SolverType = SolverType.IALSPP

class IALSModelConfig:
    def __init__(
        self,
        K: int,
        alpha0: float,
        reg: float,
        nu: float,
        init_stdev: float,
        random_seed: int,
        loss_type: LossType,
        lambda_user_feature: float,
        lambda_item_feature: float,
        feature_warmup_epochs: int,
    ) -> None: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(
        self,
        arg: tuple[int, float, float, float, float, int, LossType, float, float, int],
        /,
    ) -> None: ...

class IALSModelConfigBuilder:
    def __init__(self) -> None: ...
    def build(self) -> IALSModelConfig: ...
    def set_K(self, K: int) -> IALSModelConfigBuilder: ...
    def set_alpha0(self, alpha0: float) -> IALSModelConfigBuilder: ...
    def set_reg(self, reg: float) -> IALSModelConfigBuilder: ...
    def set_nu(self, nu: float) -> IALSModelConfigBuilder: ...
    def set_init_stdev(self, init_stdev: float) -> IALSModelConfigBuilder: ...
    def set_random_seed(self, random_seed: int) -> IALSModelConfigBuilder: ...
    def set_loss_type(self, loss_type: LossType) -> IALSModelConfigBuilder: ...
    def set_lambda_user_feature(self, value: float) -> IALSModelConfigBuilder: ...
    def set_lambda_item_feature(self, value: float) -> IALSModelConfigBuilder: ...
    def set_feature_warmup_epochs(self, value: int) -> IALSModelConfigBuilder: ...

class IALSSolverConfig:
    def __init__(
        self,
        n_threads: int,
        solver_type: SolverType,
        max_cg_steps: int,
        ialspp_subspace_dimension: int,
        ialspp_iteration: int,
    ) -> None: ...
    def __getstate__(self) -> tuple[int, SolverType, int, int, int]: ...
    def __setstate__(self, arg: tuple[int, SolverType, int, int, int], /) -> None: ...

class IALSSolverConfigBuilder:
    def __init__(self) -> None: ...
    def build(self) -> IALSSolverConfig: ...
    def set_n_threads(self, n_threads: int) -> IALSSolverConfigBuilder: ...
    def set_solver_type(self, solver_type: SolverType) -> IALSSolverConfigBuilder: ...
    def set_max_cg_steps(self, max_cg_steps: int) -> IALSSolverConfigBuilder: ...
    def set_ialspp_subspace_dimension(
        self, ialspp_subspace_dimension: int
    ) -> IALSSolverConfigBuilder: ...
    def set_ialspp_iteration(
        self, ialspp_iteration: int
    ) -> IALSSolverConfigBuilder: ...

class IALSTrainer:
    @overload
    def __init__(
        self, model_config: IALSModelConfig, interaction: scipy.sparse.csr_matrix[float]
    ) -> None: ...
    @overload
    def __init__(
        self,
        model_config: IALSModelConfig,
        interaction: scipy.sparse.csr_matrix[float],
        user_feature: scipy.sparse.csr_matrix[float]
        | Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")],
        item_feature: scipy.sparse.csr_matrix[float]
        | Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")],
    ) -> None: ...
    def step(self, solver_config: IALSSolverConfig) -> None: ...
    def user_scores(
        self, begin: int, end: int, solver_config: IALSSolverConfig
    ) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")]: ...
    def transform_user(
        self,
        interaction: scipy.sparse.csr_matrix[float],
        solver_config: IALSSolverConfig,
    ) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")]: ...
    def transform_item(
        self,
        interaction: scipy.sparse.csr_matrix[float],
        solver_config: IALSSolverConfig,
    ) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")]: ...
    def transform_user_with_feature(
        self,
        interaction: scipy.sparse.csr_matrix[float],
        feature: scipy.sparse.csr_matrix[float]
        | Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")],
        solver_config: IALSSolverConfig,
    ) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")]: ...
    def transform_item_with_feature(
        self,
        interaction: scipy.sparse.csr_matrix[float],
        feature: scipy.sparse.csr_matrix[float]
        | Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")],
        solver_config: IALSSolverConfig,
    ) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")]: ...
    def transform_user_feature(
        self,
        feature: scipy.sparse.csr_matrix[float]
        | Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")],
    ) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")]: ...
    def transform_item_feature(
        self,
        feature: scipy.sparse.csr_matrix[float]
        | Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")],
    ) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")]: ...
    def compute_loss(self, solver_config: IALSSolverConfig) -> float: ...
    @property
    def user(
        self,
    ) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")]: ...
    @user.setter
    def user(
        self,
        arg: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")],
        /,
    ) -> None: ...
    @property
    def item(
        self,
    ) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")]: ...
    @item.setter
    def item(
        self,
        arg: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")],
        /,
    ) -> None: ...
    @property
    def user_feature_weight(
        self,
    ) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")]: ...
    @user_feature_weight.setter
    def user_feature_weight(
        self,
        arg: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")],
        /,
    ) -> None: ...
    @property
    def item_feature_weight(
        self,
    ) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")]: ...
    @item_feature_weight.setter
    def item_feature_weight(
        self,
        arg: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C")],
        /,
    ) -> None: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg: tuple, /) -> None: ...
