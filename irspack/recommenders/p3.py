from typing import Optional

from irspack.definitions import InteractionMatrix
from irspack.recommenders._knn import P3alphaComputer
from irspack.recommenders.base import BaseSimilarityRecommender
from irspack.utils import get_n_threads, l1_normalize_row

from .base import BaseSimilarityRecommender, RecommenderConfig


class P3alphaConfig(RecommenderConfig):
    alpha: float = 1
    top_k: Optional[int] = None
    normalize_weight: bool = False
    n_threads: Optional[int] = None


class P3alphaRecommender(BaseSimilarityRecommender):
    """Recommendation with 3-steps random walk, proposed in

        - `Random Walks in Recommender Systems: Exact Computation and Simulations
          <https://nms.kcl.ac.uk/colin.cooper/papers/recommender-rw.pdf>`_

    The version here implements its view as KNN-based method, as pointed out in

        - `A Troubling Analysis of Reproducibility and Progress in Recommender Systems Research
          <https://arxiv.org/abs/1911.07698>`_


    Args:
        X_train_all (Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
            Input interaction matrix.
        alpha (float, optional): The power to which ``X_train_all`` is exponentiated.
            Defaults to 1. Note that this has no effect if all the entries in
            ``X_train_all`` are equal.
        top_k (Optional[int], optional): Maximal number of non-zero entries retained
            for each column of the similarity matrix ``W``.
        normalize_weight (bool, optional): Whether to perform row-wise normalization of ``W``.
            Defaults to False.
        n_threads (Optional[int], optional): Specifies the number of threads to use for the computation.
            If ``None``, the environment variable ``"IRSPACK_NUM_THREADS_DEFAULT"`` will be looked up,
            and if the variable is not set, it will be set to ``os.cpu_count()``. Defaults to None.
    """

    config_class = P3alphaConfig

    def __init__(
        self,
        X_train_all: InteractionMatrix,
        alpha: float = 1,
        top_k: Optional[int] = None,
        normalize_weight: bool = False,
        n_threads: Optional[int] = None,
    ):
        """"""
        super().__init__(X_train_all)
        self.alpha = alpha
        self.top_k = top_k
        self.normalize_weight = normalize_weight
        self.n_threads = get_n_threads(n_threads)

    def _learn(self) -> None:
        computer = P3alphaComputer(
            self.X_train_all.T,
            alpha=self.alpha,
            n_threads=self.n_threads,
        )
        top_k = self.X_train_all.shape[1] if self.top_k is None else self.top_k
        self.W_ = computer.compute_W(self.X_train_all.T, top_k)
        if self.normalize_weight:
            self.W_ = l1_normalize_row(self.W_)
