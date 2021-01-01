import gc

import numpy as np
from scipy import linalg

from ..definitions import InteractionMatrix
from .base import BaseSimilarityRecommender


class DenseSLIMRecommender(BaseSimilarityRecommender):
    """Implementation of DenseSLIM or Embarrassingly Shallow AutoEncoder (EASE ^R).

    See:

        - `Embarrassingly Shallow Autoencoders for Sparse Data
          <https://arxiv.org/abs/1905.03375>`_

    Args:
        X_train_all (Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
            Input interaction matrix.

        reg (float, optional):
            The regularization parameter for ease. Defaults to 1.0.
    """

    def __init__(self, X_train_all: InteractionMatrix, reg: float = 1):

        super(DenseSLIMRecommender, self).__init__(X_train_all)
        self.reg = reg

    def _learn(self) -> None:
        X_train_all_f32 = self.X_train_all.astype(np.float32)
        P = X_train_all_f32.T.dot(X_train_all_f32)
        P_dense: np.ndarray = P.todense()
        del P
        P_dense[np.arange(self.n_items), np.arange(self.n_items)] += self.reg
        gc.collect()
        P_dense = linalg.inv(P_dense, overwrite_a=True)

        gc.collect()
        diag_P_inv = 1 / np.diag(P_dense)
        P_dense *= -diag_P_inv[np.newaxis, :]
        range_ = np.arange(self.n_items)
        P_dense[range_, range_] = 0
        self.W_ = P_dense
