m: int
n: int
from numpy import float32

"""irspack's core module for "IALSRecommender".
Built to use
	SSE, SSE2"""
import irspack.recommenders._ials
import typing
import numpy
import scipy.sparse

_Shape = typing.Tuple[int, ...]

__all__ = [
    "IALSLearningConfig",
    "IALSLearningConfigBuilder",
    "IALSPP",
    "IALSTrainer",
    "LossType",
    "Original",
]

class IALSLearningConfig:
    def __getstate__(self) -> tuple: ...
    def __init__(
        self,
        arg0: int,
        arg1: float,
        arg2: float,
        arg3: float,
        arg4: float,
        arg5: int,
        arg6: int,
        arg7: bool,
        arg8: int,
        arg9: LossType,
    ) -> None: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    pass

class IALSLearningConfigBuilder:
    def __init__(self) -> None: ...
    def build(self) -> IALSLearningConfig: ...
    def set_K(self, arg0: int) -> IALSLearningConfigBuilder: ...
    def set_alpha0(self, arg0: float) -> IALSLearningConfigBuilder: ...
    def set_init_stdev(self, arg0: float) -> IALSLearningConfigBuilder: ...
    def set_loss_type(self, arg0: LossType) -> IALSLearningConfigBuilder: ...
    def set_max_cg_steps(self, arg0: int) -> IALSLearningConfigBuilder: ...
    def set_n_threads(self, arg0: int) -> IALSLearningConfigBuilder: ...
    def set_nu(self, arg0: float) -> IALSLearningConfigBuilder: ...
    def set_random_seed(self, arg0: int) -> IALSLearningConfigBuilder: ...
    def set_reg(self, arg0: float) -> IALSLearningConfigBuilder: ...
    def set_use_cg(self, arg0: bool) -> IALSLearningConfigBuilder: ...
    pass

class IALSTrainer:
    def __getstate__(self) -> tuple: ...
    def __init__(
        self, arg0: IALSLearningConfig, arg1: scipy.sparse.csr_matrix[numpy.float32]
    ) -> None: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def step(self) -> None: ...
    def transform_item(
        self, arg0: scipy.sparse.csr_matrix[numpy.float32]
    ) -> numpy.ndarray[numpy.float32, _Shape[m, n]]: ...
    def transform_user(
        self, arg0: scipy.sparse.csr_matrix[numpy.float32]
    ) -> numpy.ndarray[numpy.float32, _Shape[m, n]]: ...
    def user_scores(
        self, arg0: int, arg1: int
    ) -> numpy.ndarray[numpy.float32, _Shape[m, n]]: ...
    @property
    def item(self) -> numpy.ndarray[numpy.float32, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float32, _Shape[m, n]]
        """
    @item.setter
    def item(self, arg0: numpy.ndarray[numpy.float32, _Shape[m, n]]) -> None:
        pass
    @property
    def user(self) -> numpy.ndarray[numpy.float32, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float32, _Shape[m, n]]
        """
    @user.setter
    def user(self, arg0: numpy.ndarray[numpy.float32, _Shape[m, n]]) -> None:
        pass
    pass

class LossType:
    """
    Members:

      Original

      IALSPP
    """

    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    IALSPP: irspack.recommenders._ials.LossType  # value = <LossType.IALSPP: 1>
    Original: irspack.recommenders._ials.LossType  # value = <LossType.Original: 0>
    __members__: dict  # value = {'Original': <LossType.Original: 0>, 'IALSPP': <LossType.IALSPP: 1>}
    pass

IALSPP: irspack.recommenders._ials.LossType  # value = <LossType.IALSPP: 1>
Original: irspack.recommenders._ials.LossType  # value = <LossType.Original: 0>
