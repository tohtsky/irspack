from typing import Optional

from scipy import sparse as sps
from sklearn.linear_model import ElasticNet

from irspack.definitions import InteractionMatrix
from irspack.recommenders.base import BaseSimilarityRecommender
from irspack.utils import get_n_threads
from irspack.utils._util_cpp import (
    slim_weight_allow_negative,
    slim_weight_positive_only,
)


class SLIMRecommender(BaseSimilarityRecommender):
    def __init__(
        self,
        X_train_all: InteractionMatrix,
        alpha: float = 0.05,
        l1_ratio: float = 0.01,
        positive_only: bool = True,
        n_threads: Optional[int] = None,
        n_iter: int = 5,
    ):
        super().__init__(X_train_all)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.n_threads = get_n_threads(n_threads)
        self.n_iter = n_iter

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
            )
        else:
            self.W_ = slim_weight_allow_negative(
                self.X_train_all,
                n_threads=self.n_threads,
                n_iter=self.n_iter,
                l2_coeff=l2_coeff,
                l1_coeff=l1_coeff,
            )
