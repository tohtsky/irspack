from typing import Optional

from sklearn.preprocessing import normalize

from ..definitions import InteractionMatrix
from ._knn import P3alphaComputer
from .base import BaseRecommenderWithThreadingSupport, BaseSimilarityRecommender


class P3alphaRecommender(
    BaseSimilarityRecommender, BaseRecommenderWithThreadingSupport
):
    """Recommendation with 3-steps random walk:

        - `Random Walks in Recommender Systems: Exact Computation and Simulations
          <https://nms.kcl.ac.uk/colin.cooper/papers/recommender-rw.pdf>`_

    The version here also implements its view as KNN-based method, as pointed out in

        - `A Troubling Analysis of Reproducibility and Progress in Recommender Systems Research
          <https://arxiv.org/abs/1911.07698>`_


    Args:
        X_train_all (Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
            Input interaction matrix.
        alpha (float, optional): The power to which ``X_train_all`` are exponentiated.
            Defaults to 1.
        top_k (Optional[int], optional): Maximal number of non-zero entries retained
            for each column of the similarity matrix ``W``.
        normalize_weight (bool, optional): Whether to normalize ``W`` into row-direction.
            Defaults to False.
        n_thread (Optional[int], optional): The number of threads to be used for computation.
            Defaults to 1.
    """

    def __init__(
        self,
        X_train_all: InteractionMatrix,
        alpha: float = 1,
        top_k: Optional[int] = None,
        normalize_weight: bool = False,
        n_thread: Optional[int] = 1,
    ):
        """"""
        super().__init__(X_train_all, n_thread=n_thread)
        self.alpha = alpha
        self.top_k = top_k
        self.normalize_weight = normalize_weight

    def _learn(self) -> None:
        computer = P3alphaComputer(
            self.X_train_all.T,
            alpha=self.alpha,
            n_thread=self.n_thread,
        )
        top_k = self.X_train_all.shape[1] if self.top_k is None else self.top_k
        self.W_ = computer.compute_W(self.X_train_all.T, top_k)
        if self.normalize_weight:
            self.W_ = normalize(self.W_, norm="l1", axis=1)
