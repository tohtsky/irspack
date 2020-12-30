from scipy import sparse as sps
from sklearn.linear_model import ElasticNet

from ..definitions import InteractionMatrix
from .base import BaseSimilarityRecommender


def slim_weight(X: InteractionMatrix, alpha: float, l1_ratio: float) -> sps.csr_matrix:
    model = ElasticNet(
        fit_intercept=False,
        positive=True,
        copy_X=False,
        precompute=True,
        selection="random",
        max_iter=100,
        tol=1e-4,
        alpha=alpha,
        l1_ratio=l1_ratio,
    )
    coeff_all = []
    A: sps.csc_matrix = X.tocsc()
    for i in range(X.shape[1]):
        if i % 1000 == 0:
            print(f"Slim Iteration: {i}")
        start_pos = int(A.indptr[i])
        end_pos = int(A.indptr[i + 1])
        current_item_data_backup = A.data[start_pos:end_pos].copy()
        target = A[:, i].toarray().ravel()
        A.data[start_pos:end_pos] = 0.0
        model.fit(A, target)
        coeff_all.append(model.sparse_coef_)
        A.data[start_pos:end_pos] = current_item_data_backup
    return sps.vstack(coeff_all, format="csr")


class SLIMRecommender(BaseSimilarityRecommender):
    def __init__(
        self,
        X_train_all: InteractionMatrix,
        alpha: float = 0.05,
        l1_ratio: float = 0.01,
    ):
        super(SLIMRecommender, self).__init__(X_train_all)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def _learn(self) -> None:
        self.W_ = slim_weight(
            self.X_train_all, alpha=self.alpha, l1_ratio=self.l1_ratio
        )
