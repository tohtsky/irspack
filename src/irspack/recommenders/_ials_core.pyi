import enum
from typing import Annotated

import scipy.sparse
from numpy.typing import ArrayLike

class LossType(enum.Enum):
    ORIGINAL = 0

    IALSPP = 1

ORIGINAL: LossType = LossType.ORIGINAL

IALSPP: SolverType = SolverType.IALSPP

class SolverType(enum.Enum):
    CHOLESKY = 0

    CG = 1

    IALSPP = 2

CHOLESKY: SolverType = SolverType.CHOLESKY

CG: SolverType = SolverType.CG

class IALSModelConfig:
    def __init__(
        self,
        arg0: int,
        arg1: float,
        arg2: float,
        arg3: float,
        arg4: float,
        arg5: int,
        arg6: LossType,
        /,
    ) -> None: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(
        self, arg: tuple[int, float, float, float, float, int, LossType], /
    ) -> None: ...

class IALSModelConfigBuilder:
    def __init__(self) -> None: ...
    def build(self) -> IALSModelConfig: ...
    def set_K(self, arg: int, /) -> IALSModelConfigBuilder: ...
    def set_alpha0(self, arg: float, /) -> IALSModelConfigBuilder: ...
    def set_reg(self, arg: float, /) -> IALSModelConfigBuilder: ...
    def set_nu(self, arg: float, /) -> IALSModelConfigBuilder: ...
    def set_init_stdev(self, arg: float, /) -> IALSModelConfigBuilder: ...
    def set_random_seed(self, arg: int, /) -> IALSModelConfigBuilder: ...
    def set_loss_type(self, arg: LossType, /) -> IALSModelConfigBuilder: ...

class IALSSolverConfig:
    def __init__(
        self, arg0: int, arg1: SolverType, arg2: int, arg3: int, arg4: int, /
    ) -> None: ...
    def __getstate__(self) -> tuple[int, SolverType, int, int, int]: ...
    def __setstate__(self, arg: tuple[int, SolverType, int, int, int], /) -> None: ...

class IALSSolverConfigBuilder:
    def __init__(self) -> None: ...
    def build(self) -> IALSSolverConfig: ...
    def set_n_threads(self, arg: int, /) -> IALSSolverConfigBuilder: ...
    def set_solver_type(self, arg: SolverType, /) -> IALSSolverConfigBuilder: ...
    def set_max_cg_steps(self, arg: int, /) -> IALSSolverConfigBuilder: ...
    def set_ialspp_subspace_dimension(self, arg: int, /) -> IALSSolverConfigBuilder: ...
    def set_ialspp_iteration(self, arg: int, /) -> IALSSolverConfigBuilder: ...

class IALSTrainer:
    def __init__(
        self, arg0: IALSModelConfig, arg1: scipy.sparse.csr_matrix[float], /
    ) -> None: ...
    def step(self, arg: IALSSolverConfig, /) -> None: ...
    def user_scores(
        self, arg0: int, arg1: int, arg2: IALSSolverConfig, /
    ) -> Annotated[ArrayLike, dict(dtype="float32", shape=(None, None), order="C")]: ...
    def transform_user(
        self, arg0: scipy.sparse.csr_matrix[float], arg1: IALSSolverConfig, /
    ) -> Annotated[ArrayLike, dict(dtype="float32", shape=(None, None), order="C")]: ...
    def transform_item(
        self, arg0: scipy.sparse.csr_matrix[float], arg1: IALSSolverConfig, /
    ) -> Annotated[ArrayLike, dict(dtype="float32", shape=(None, None), order="C")]: ...
    def compute_loss(self, arg: IALSSolverConfig, /) -> float: ...
    @property
    def user(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float32", shape=(None, None), order="C")]: ...
    @user.setter
    def user(
        self,
        arg: Annotated[ArrayLike, dict(dtype="float32", shape=(None, None), order="C")],
        /,
    ) -> None: ...
    @property
    def item(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float32", shape=(None, None), order="C")]: ...
    @item.setter
    def item(
        self,
        arg: Annotated[ArrayLike, dict(dtype="float32", shape=(None, None), order="C")],
        /,
    ) -> None: ...
    def __getstate__(
        self,
    ) -> tuple[
        IALSModelConfig,
        Annotated[ArrayLike, dict(dtype="float32", shape=(None, None), order="C")],
        Annotated[ArrayLike, dict(dtype="float32", shape=(None, None), order="C")],
    ]: ...
    def __setstate__(
        self,
        arg: tuple[
            IALSModelConfig,
            Annotated[ArrayLike, dict(dtype="float32", shape=(None, None), order="C")],
            Annotated[ArrayLike, dict(dtype="float32", shape=(None, None), order="C")],
        ],
        /,
    ) -> None: ...
