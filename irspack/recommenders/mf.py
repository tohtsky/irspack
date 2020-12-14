from .base import BaseRecommender
from ..definitions import InteractionMatrix, UserIndexArray
from scipy import sparse as sps
from sklearn.utils.extmath import randomized_svd


class SVDRecommender(BaseRecommender):
    def __init__(self, X_all: InteractionMatrix, n_components: int = 128):
        super().__init__(X_all)
        U, diag, V = randomized_svd(self.X_all, n_components=n_components)
        self.U = U
        self.V = sps.diags(diag) * V

    def get_score(self, user_indices: UserIndexArray):
        return self.U[user_indices].dot(self.V)
