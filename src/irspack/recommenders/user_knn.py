from abc import abstractmethod
from typing import Optional, Union

from .._threading import get_n_threads
from ..definitions import InteractionMatrix
from ..optimization.parameter_range import (
    CategoricalRange,
    UniformFloatRange,
    default_tune_range_knn_with_weighting,
)
from ..utils import okapi_BM_25_weight, remove_diagonal, tf_idf_weight
from ._knn import (
    AsymmetricSimilarityComputer,
    CosineSimilarityComputer,
    JaccardSimilarityComputer,
    TverskyIndexComputer,
)
from .base import BaseUserSimilarityRecommender, RecommenderConfig
from .knn import FeatureWeightingScheme


class BaseUserKNNConfig(RecommenderConfig):
    shrinkage: float = 0.0
    top_k: int = 100
    n_threads: Optional[int] = None
    feature_weighting: str = "NONE"
    bm25_k1: float = 1.2
    bm25_b: float = 0.75


class BaseUserKNNRecommender(BaseUserSimilarityRecommender):
    def __init__(
        self,
        X_train_all: InteractionMatrix,
        shrinkage: float = 0.0,
        top_k: int = 100,
        n_threads: Optional[int] = None,
        feature_weighting: str = "NONE",
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
    ):
        super().__init__(X_train_all)
        self.shrinkage = shrinkage
        self.top_k = top_k
        self.feature_weighting = FeatureWeightingScheme(feature_weighting)
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.n_threads = get_n_threads(n_threads)

    @abstractmethod
    def _create_computer(self, X: InteractionMatrix) -> Union[
        CosineSimilarityComputer,
        AsymmetricSimilarityComputer,
        JaccardSimilarityComputer,
        TverskyIndexComputer,
    ]:
        raise NotImplementedError("")

    def _learn(self) -> None:
        if self.feature_weighting == FeatureWeightingScheme.NONE:
            X_weighted = self.X_train_all
        elif self.feature_weighting == FeatureWeightingScheme.TF_IDF:
            X_weighted = tf_idf_weight(self.X_train_all)
        elif self.feature_weighting == FeatureWeightingScheme.BM_25:
            X_weighted = okapi_BM_25_weight(self.X_train_all, self.bm25_k1, self.bm25_b)
        else:
            raise RuntimeError("Unknown weighting scheme.")

        computer = self._create_computer(X_weighted)
        self.U_ = remove_diagonal(
            computer.compute_similarity(self.X_train_all, self.top_k)
        )


class CosineUserKNNConfig(BaseUserKNNConfig):
    normalize: bool = True


class CosineUserKNNRecommender(BaseUserKNNRecommender):
    r"""K-nearest neighbor recommender system based on cosine similarity. That is, the similarity matrix ``U`` is given by (row-wise top-k restricted)

    .. math::

        \mathrm{U}_{u,v} = \begin{cases}
            \frac{\sum_{i} X_{ui} X_{vi}}{||X_{u*}||_2 ||X_{v*}||_2 + \mathrm{shrinkage}} & (\text{if normalize = True}) \\
            \sum_{i} X_{ui} X_{vi} & (\text{if normalize = False})
        \end{cases}


    Args:
        X_train_all (Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
            Input interaction matrix.
        shrinkage (float, optional):
            The shrinkage parameter for regularization. Defaults to 0.0.
        normalize (bool, optional):
            Whether to normalize the similarity. Defaults to False.
        top_k (int, optional):
            Specifies the maximal number of allowed neighbors. Defaults to 100.
        feature_weighting (str, optional):
            Specifies how to weight the feature. Must be one of:

                - "NONE" : no feature weighting
                - "TF_IDF" : TF-IDF weighting
                - "BM_25" : `Okapi BM-25 weighting <https://en.wikipedia.org/wiki/Okapi_BM25>`_

            Defaults to "NONE".
        bm25_k1 (float, optional):
            The k1 parameter for BM25. Ignored if ``feature_weighting`` is not "BM_25". Defaults to 1.2.
        bm25_b (float, optional):
            The b parameter for BM25. Ignored if ``feature_weighting`` is not "BM_25". Defaults to 0.75.
        n_threads (Optional[int], optional): Specifies the number of threads to use for the computation.
            If ``None``, the environment variable ``"IRSPACK_NUM_THREADS_DEFAULT"`` will be looked up,
            and if the variable is not set, it will be set to ``os.cpu_count()``. Defaults to None.
    """

    config_class = CosineUserKNNConfig
    default_tune_range = default_tune_range_knn_with_weighting.copy() + [
        CategoricalRange("normalize", [False, True])
    ]

    def __init__(
        self,
        X_train_all: InteractionMatrix,
        shrinkage: float = 0.0,
        normalize: bool = True,
        top_k: int = 100,
        feature_weighting: str = "NONE",
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        n_threads: Optional[int] = None,
    ):
        super().__init__(
            X_train_all,
            shrinkage,
            top_k,
            n_threads,
            feature_weighting=feature_weighting,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )
        self.normalize = normalize

    def _create_computer(self, X: InteractionMatrix) -> CosineSimilarityComputer:
        return CosineSimilarityComputer(
            X, self.shrinkage, self.normalize, self.n_threads
        )


class AsymmetricCosineUserKNNConfig(BaseUserKNNConfig):
    alpha: float = 0.5


class AsymmetricCosineUserKNNRecommender(BaseUserKNNRecommender):
    r"""K-nearest neighbor recommender system based on asymmetric cosine similarity. That is, the similarity matrix ``U`` is given by (row-wise top-k restricted)

    .. math::

        \mathrm{U}_{u,v} = \frac{\sum_{i} X_{ui} X_{vi}}{||X_{u*}||^{2\alpha}_2 ||X_{v*}||^{2(1-\alpha)}_2 + \mathrm{shrinkage}}

    Args:
        X_train_all (Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
            Input interaction matrix.
        shrinkage (float, optional):
            The shrinkage parameter for regularization. Defaults to 0.0.
        alpha (bool, optional):
            Specifies :math:`\alpha`. Defaults to 0.5.
        top_k (int, optional):
            Specifies the maximal number of allowed neighbors. Defaults to 100.
        feature_weighting (str, optional):
            Specifies how to weight the feature. Must be one of:

                - "NONE" : no feature weighting
                - "TF_IDF" : TF-IDF weighting
                - "BM_25" : `Okapi BM-25 weighting <https://en.wikipedia.org/wiki/Okapi_BM25>`_

            Defaults to "NONE".
        bm25_k1 (float, optional):
            The k1 parameter for BM25. Ignored if ``feature_weighting`` is not "BM_25". Defaults to 1.2.
        bm25_b (float, optional):
            The b parameter for BM25. Ignored if ``feature_weighting`` is not "BM_25". Defaults to 0.75.
        n_threads (Optional[int], optional): Specifies the number of threads to use for the computation.
            If ``None``, the environment variable ``"IRSPACK_NUM_THREADS_DEFAULT"`` will be looked up,
            and if the variable is not set, it will be set to ``os.cpu_count()``. Defaults to None.
    """

    config_class = AsymmetricCosineUserKNNConfig
    default_tune_range = default_tune_range_knn_with_weighting + [
        UniformFloatRange("alpha", 0, 1)
    ]

    def __init__(
        self,
        X_train_all: InteractionMatrix,
        shrinkage: float = 0.0,
        alpha: float = 0.5,
        top_k: int = 100,
        feature_weighting: str = "NONE",
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        n_threads: Optional[int] = None,
    ):
        super().__init__(
            X_train_all,
            shrinkage,
            top_k,
            n_threads,
            feature_weighting=feature_weighting,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )
        self.alpha = alpha

    def _create_computer(self, X: InteractionMatrix) -> AsymmetricSimilarityComputer:
        return AsymmetricSimilarityComputer(
            X, self.shrinkage, self.alpha, self.n_threads
        )
