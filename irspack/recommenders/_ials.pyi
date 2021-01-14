m: int
n: int
from numpy import float32

"""irspack's core module for "IALSRecommender".
Built to use
	SSE, SSE2"""
import irspack.recommenders._ials
from typing import *
from typing import Iterable as iterable
from typing import Iterator as iterator
from numpy import float64

_Shape = Tuple[int, ...]
import numpy
import scipy.sparse

__all__ = ["IALSLearningConfig", "IALSLearningConfigBuilder", "IALSTrainer"]

class IALSLearningConfig:
    def __getstate__(self) -> tuple: ...
    def __init__(
        self,
        arg0: int,
        arg1: float,
        arg2: float,
        arg3: float,
        arg4: int,
        arg5: int,
        arg6: bool,
        arg7: int,
    ) -> None: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    pass

class IALSLearningConfigBuilder:
    def __init__(self) -> None: ...
    def build(self) -> IALSLearningConfig: ...
    def set_K(self, arg0: int) -> IALSLearningConfigBuilder: ...
    def set_alpha(self, arg0: float) -> IALSLearningConfigBuilder: ...
    def set_init_stdev(self, arg0: float) -> IALSLearningConfigBuilder: ...
    def set_max_cg_steps(self, arg0: int) -> IALSLearningConfigBuilder: ...
    def set_n_threads(self, arg0: int) -> IALSLearningConfigBuilder: ...
    def set_random_seed(self, arg0: int) -> IALSLearningConfigBuilder: ...
    def set_reg(self, arg0: float) -> IALSLearningConfigBuilder: ...
    def set_use_cg(self, arg0: bool) -> IALSLearningConfigBuilder: ...
    pass

class IALSTrainer:
    def __getstate__(self) -> tuple: ...
    def __init__(
        self, arg0: IALSLearningConfig, arg1: scipy.sparse.csr_matrix[float32]
    ) -> None: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def step(self) -> None: ...
    def transform_item(
        self, arg0: scipy.sparse.csr_matrix[float32]
    ) -> numpy.ndarray[float32, _Shape[m, n]]: ...
    def transform_user(
        self, arg0: scipy.sparse.csr_matrix[float32]
    ) -> numpy.ndarray[float32, _Shape[m, n]]: ...
    def user_scores(
        self, arg0: int, arg1: int
    ) -> numpy.ndarray[float32, _Shape[m, n]]: ...
    @property
    def item(self) -> numpy.ndarray[float32, _Shape[m, n]]:
        """
        :type: numpy.ndarray[float32, _Shape[m, n]]
        """
    @item.setter
    def item(self, arg0: numpy.ndarray[float32, _Shape[m, n]]) -> None:
        pass
    @property
    def user(self) -> numpy.ndarray[float32, _Shape[m, n]]:
        """
        :type: numpy.ndarray[float32, _Shape[m, n]]
        """
    @user.setter
    def user(self, arg0: numpy.ndarray[float32, _Shape[m, n]]) -> None:
        pass
    pass
