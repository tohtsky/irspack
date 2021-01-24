from typing import Optional

from irspack.definitions import InteractionMatrix
from irspack.recommenders.base import BaseSimilarityRecommender
from irspack.utils import get_n_threads
from irspack.utils._util_cpp import (
    slim_weight_allow_negative,
    slim_weight_positive_only,
)


class SLIMRecommender(BaseSimilarityRecommender):
    """`SLIM <https://dl.acm.org/doi/10.1109/ICDM.2011.134>`_ with ElasticNet-type loss function:

    .. math ::

        \mathrm{loss} = \\frac{1}{2} ||X - XB|| ^2 _F + \\frac{\\alpha (1 - l_1)  U}{2} ||B|| ^2 _FF + \\alpha l_1  U |B|

    The implementation relies on a simple (parallelized) cyclic-coordinate descent method.

    Args:
        X_train_all:
            Input interaction matrix.
        alpha:
            Determines the strength of L1/L2 regularization (see above). Defaults to 0.05.
        l1_ratio:
            Determines the strength of L1 regularization relative to alpha. Defaults to 0.01.
        positive_only:
            Whether we constrain the weight matrix to be non-negative. Defaults to True.
        n_iter:
            The number of coordinate-descent iterations. Defaults to 100.
        tol:
            Tolerance parameter for cd iteration, i.e., if the maximal parameter change
            of the coordinate-descent single iteration is smaller than this value,
            the iteration will terminate. Defaults to 1e-4.
        n_threads:
            Specifies the number of threads to use for the computation.
            If ``None``, the environment variable ``"IRSPACK_NUM_THREADS_DEFAULT"`` will be looked up,
            and if there is no such an environment variable, it will be set to 1. Defaults to None.
    """

    def __init__(
        self,
        X_train_all: InteractionMatrix,
        alpha: float = 0.05,
        l1_ratio: float = 0.01,
        positive_only: bool = True,
        n_iter: int = 100,
        tol: float = 1e-4,
        n_threads: Optional[int] = None,
    ):
        super().__init__(X_train_all)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.n_threads = get_n_threads(n_threads)
        self.n_iter = n_iter
        self.tol = tol

    def _learn(self) -> None:
        l2_coeff = self.n_users * self.alpha * (1 - self.l1_ratio)
        l1_coeff = self.n_users * self.alpha * self.l1_ratio

        if self.positive_only:
            self.W_ = slim_weight_positive_only(
                self.X_train_all,
                n_threads=self.n_threads,
                n_iter=self.n_iter,
                l2_coeff=l2_coeff,
                l1_coeff=l1_coeff,
                tol=self.tol,
            )
        else:
            self.W_ = slim_weight_allow_negative(
                self.X_train_all,
                n_threads=self.n_threads,
                n_iter=self.n_iter,
                l2_coeff=l2_coeff,
                l1_coeff=l1_coeff,
                tol=self.tol,
            )
