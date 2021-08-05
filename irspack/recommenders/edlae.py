import gc

import numpy as np
from scipy import linalg, sparse

from ..definitions import InteractionMatrix
from .base import BaseSimilarityRecommender, RecommenderConfig


class EDLAEConfig(RecommenderConfig):
    reg: float = 1.0
    dropout_p: float = 0.1


class EDLAERecommender(BaseSimilarityRecommender):
    """Implementation of EDLAE (Emphasized Denoising Linear Autoencoder).

    See:
        - `Autoencoders that don't overfit towards the Identity (NeurIPS 2020)`
          by Harald Steck (NetFlix)

    Args:
        X_train_all (Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
            Input interaction matrix.

        reg (float, optional):
            The L2 constant regularization parameter. Defaults to 1.0.

        dropout_p (float, optional):
            Probability of dropout. Defaults to 0.1
    """

    config_class = EDLAEConfig

    def __init__(
        self, X_train_all: InteractionMatrix, reg: float = 1.0, dropout_p: float = 0.1
    ):

        super(EDLAERecommender, self).__init__(X_train_all)
        self.reg = reg
        self.dropout_p = dropout_p

    def _learn(self) -> None:
        X_train_all_f32 = self.X_train_all.astype(np.float32)

        P = X_train_all_f32.T.dot(X_train_all_f32)
        P_dense: np.ndarray = P.todense()
        del P
        q = 1 - self.dropout_p
        lamb = self.dropout_p / q * np.diag(P_dense) + self.reg
        P_dense[np.arange(self.n_items), np.arange(self.n_items)] += lamb
        gc.collect()
        P_dense = linalg.inv(P_dense, overwrite_a=True)

        gc.collect()
        diag_P_inv = 1 / np.diag(P_dense)
        P_dense *= -diag_P_inv[np.newaxis, :]
        range_ = np.arange(self.n_items)
        P_dense[range_, range_] = 0
        self.W_ = P_dense
