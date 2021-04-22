import warnings
from typing import List, Type

from irspack.definitions import InteractionMatrix

from ..optimizers.base_optimizer import (
    BaseOptimizer,
    BaseOptimizerWithEarlyStopping,
    LowMemoryError,
)
from ..parameter_tuning import (
    CategoricalSuggestion,
    IntegerSuggestion,
    LogUniformSuggestion,
    Suggestion,
    UniformSuggestion,
)
from ..recommenders import (
    AsymmetricCosineKNNRecommender,
    AsymmetricCosineUserKNNRecommender,
    CosineKNNRecommender,
    CosineUserKNNRecommender,
    DenseSLIMRecommender,
    IALSRecommender,
    JaccardKNNRecommender,
    NMFRecommender,
    P3alphaRecommender,
    RP3betaRecommender,
    SLIMRecommender,
    TopPopRecommender,
    TruncatedSVDRecommender,
    TverskyIndexKNNRecommender,
)

default_tune_range_knn = [
    IntegerSuggestion("top_k", 4, 1000),
    UniformSuggestion("shrinkage", 0, 1000),
]

default_tune_range_knn_with_weighting = [
    IntegerSuggestion("top_k", 4, 1000),
    UniformSuggestion("shrinkage", 0, 1000),
    CategoricalSuggestion("feature_weighting", ["NONE", "TF_IDF", "BM_25"]),
]


_BaseOptimizerArgsString = """Args:
    data (Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
        The train data.
    val_evaluator (Evaluator):
        The validation evaluator which measures the performance of the recommenders.
    logger (Optional[logging.Logger], optional) :
        The logger used during the optimization steps. Defaults to None.
        If ``None``, the default logger of irspack will be used.
    suggest_overwrite (List[Suggestion], optional) :
        Customizes (e.g. enlarging the parameter region or adding new parameters to be tuned)
        the default parameter search space defined by ``default_tune_range``
        Defaults to list().
    fixed_params (Dict[str, Any], optional):
        Fixed parameters passed to recommenders during the optimization procedure.
        If such a parameter exists in ``default_tune_range``, it will not be tuned.
        Defaults to dict().
"""

_BaseOptimizerWithEarlyStoppingArgsString = """Args:
    data (Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
        The train data.
    val_evaluator (Evaluator):
        The validation evaluator which measures the performance of the recommenders.
    logger (Optional[logging.Logger], optional):
        The logger used during the optimization steps. Defaults to None.
        If ``None``, the default logger of irspack will be used.
    suggest_overwrite (List[Suggestion], optional):
        Customizes (e.g. enlarging the parameter region or adding new parameters to be tuned)
        the default parameter search space defined by ``default_tune_range``
        Defaults to list().
    fixed_params (Dict[str, Any], optional):
        Fixed parameters passed to recommenders during the optimization procedure.
        If such a parameter exists in ``default_tune_range``, it will not be tuned.
        Defaults to dict().
    max_epoch (int, optional):
        The maximal number of epochs for the training. Defaults to 512.
    validate_epoch (int, optional):
        The frequency of validation score measurement. Defaults to 5.
    score_degradation_max (int, optional):
        Maximal number of allowed score degradation. Defaults to 5. Defaults to 5.
"""


def _get_maximal_n_components_for_budget(
    X: InteractionMatrix, budget: int, n_component_max_default: int
) -> int:
    return min(int((budget * 1e6) / (8 * (sum(X.shape) + 1))), n_component_max_default)


def _add_docstring(
    cls: Type[BaseOptimizer], args: str = _BaseOptimizerArgsString
) -> None:

    if cls.default_tune_range:

        ranges = ""
        for suggest in cls.default_tune_range:
            ranges += f"          - ``{suggest!r}``\n"

        ranges += "\n"
        tune_range = f"""The default tune range is

{ranges}"""
    else:
        tune_range = "   There is no tunable parameters."
    docs = f"""Optimizer class for :class:`irspack.recommenders.{cls.recommender_class.__name__}`.

{tune_range}

{args}

    """
    cls.__doc__ = docs


class TopPopOptimizer(BaseOptimizer):
    default_tune_range: List[Suggestion] = []
    recommender_class = TopPopRecommender

    @classmethod
    def tune_range_given_memory_budget(
        cls, X: InteractionMatrix, memory_budget: int
    ) -> List[Suggestion]:
        return []


_add_docstring(TopPopOptimizer)


class IALSOptimizer(BaseOptimizerWithEarlyStopping):
    default_tune_range = [
        IntegerSuggestion("n_components", 4, 300),
        LogUniformSuggestion("alpha", 1, 100),
        LogUniformSuggestion("reg", 1e-10, 1e-2),
    ]
    recommender_class = IALSRecommender

    @classmethod
    def tune_range_given_memory_budget(
        cls, X: InteractionMatrix, memory_budget: int
    ) -> List[Suggestion]:
        n_components = _get_maximal_n_components_for_budget(X, memory_budget, 300)
        return [
            IntegerSuggestion("n_components", 4, n_components),
        ]


_add_docstring(IALSOptimizer, _BaseOptimizerWithEarlyStoppingArgsString)


class DenseSLIMOptimizer(BaseOptimizer):
    default_tune_range: List[Suggestion] = [LogUniformSuggestion("reg", 1, 1e4)]
    recommender_class = DenseSLIMRecommender

    @classmethod
    def tune_range_given_memory_budget(
        cls, X: InteractionMatrix, memory_budget: int
    ) -> List[Suggestion]:
        n_items: int = X.shape[1]
        if memory_budget < (2 * n_items ** 2):
            raise LowMemoryError(
                f"Memory budget {memory_budget} too small for DenseSLIM to work."
            )
        return cls.default_tune_range


_add_docstring(DenseSLIMOptimizer)


class TruncatedSVDOptimizer(BaseOptimizer):
    default_tune_range = [IntegerSuggestion("n_components", 4, 512)]
    recommender_class = TruncatedSVDRecommender

    @classmethod
    def tune_range_given_memory_budget(
        cls, X: InteractionMatrix, memory_budget: int
    ) -> List[Suggestion]:
        n_components = _get_maximal_n_components_for_budget(X, memory_budget, 512)
        return [
            IntegerSuggestion("n_components", 4, n_components),
        ]


_add_docstring(TruncatedSVDOptimizer)


class NMFOptimizer(BaseOptimizer):
    default_tune_range = [
        IntegerSuggestion("n_components", 4, 512),
        LogUniformSuggestion("alpha", 1e-10, 1e-1),
        UniformSuggestion("l1_ratio", 0, 1),
    ]
    recommender_class = NMFRecommender

    @classmethod
    def tune_range_given_memory_budget(
        cls, X: InteractionMatrix, memory_budget: int
    ) -> List[Suggestion]:
        n_components = _get_maximal_n_components_for_budget(X, memory_budget, 512)
        return [
            IntegerSuggestion("top_k", 4, n_components),
        ]


_add_docstring(NMFOptimizer)


class SimilarityBasedOptimizerBase(BaseOptimizer):
    @classmethod
    def tune_range_given_memory_budget(
        cls, X: InteractionMatrix, memory_budget: int
    ) -> List[Suggestion]:
        top_k_max = min(int(1e6 * memory_budget / 4 // (X.shape[1] + 1)), 1024)
        return [
            IntegerSuggestion("top_k", 4, top_k_max),
        ]


class SLIMOptimizer(SimilarityBasedOptimizerBase):
    default_tune_range = [
        LogUniformSuggestion("alpha", 1e-5, 1),
        UniformSuggestion("l1_ratio", 0, 1),
    ]
    recommender_class = SLIMRecommender


_add_docstring(SLIMOptimizer)


class P3alphaOptimizer(SimilarityBasedOptimizerBase):
    default_tune_range = [
        IntegerSuggestion("top_k", low=10, high=1000),
        CategoricalSuggestion("normalize_weight", [True, False]),
    ]
    recommender_class = P3alphaRecommender


_add_docstring(P3alphaOptimizer)


class RP3betaOptimizer(SimilarityBasedOptimizerBase):
    default_tune_range = [
        IntegerSuggestion("top_k", 2, 1000),
        LogUniformSuggestion("beta", 1e-5, 5e-1),
        CategoricalSuggestion("normalize_weight", [True, False]),
    ]
    recommender_class = RP3betaRecommender


_add_docstring(RP3betaOptimizer)


class CosineKNNOptimizer(SimilarityBasedOptimizerBase):
    default_tune_range = default_tune_range_knn_with_weighting.copy() + [
        CategoricalSuggestion("normalize", [False, True])
    ]

    recommender_class = CosineKNNRecommender


_add_docstring(CosineKNNOptimizer)


class AsymmetricCosineKNNOptimizer(SimilarityBasedOptimizerBase):
    default_tune_range = default_tune_range_knn_with_weighting + [
        UniformSuggestion("alpha", 0, 1)
    ]

    recommender_class = AsymmetricCosineKNNRecommender


_add_docstring(AsymmetricCosineKNNOptimizer)


class JaccardKNNOptimizer(SimilarityBasedOptimizerBase):

    default_tune_range = default_tune_range_knn.copy()
    recommender_class = JaccardKNNRecommender


_add_docstring(JaccardKNNOptimizer)


class TverskyIndexKNNOptimizer(SimilarityBasedOptimizerBase):
    default_tune_range = default_tune_range_knn.copy() + [
        UniformSuggestion("alpha", 0, 2),
        UniformSuggestion("beta", 0, 2),
    ]

    recommender_class = TverskyIndexKNNRecommender


_add_docstring(TverskyIndexKNNOptimizer)


class UserSimilarityBasedOptimizerBase(BaseOptimizer):
    @classmethod
    def tune_range_given_memory_budget(
        cls, X: InteractionMatrix, memory_budget: int
    ) -> List[Suggestion]:
        top_k_max = min(int(1e6 * memory_budget / 4 // (X.shape[0] + 1)), 1024)
        return [
            IntegerSuggestion("top_k", 4, top_k_max),
        ]


class CosineUserKNNOptimizer(UserSimilarityBasedOptimizerBase):
    default_tune_range = default_tune_range_knn_with_weighting.copy() + [
        CategoricalSuggestion("normalize", [False, True])
    ]

    recommender_class = CosineUserKNNRecommender


_add_docstring(CosineUserKNNOptimizer)


class AsymmetricCosineUserKNNOptimizer(UserSimilarityBasedOptimizerBase):
    default_tune_range = default_tune_range_knn_with_weighting + [
        UniformSuggestion("alpha", 0, 1)
    ]

    recommender_class = AsymmetricCosineUserKNNRecommender


_add_docstring(AsymmetricCosineUserKNNOptimizer)


try:
    from ..recommenders.bpr import BPRFMRecommender

    class BPRFMOptimizer(BaseOptimizerWithEarlyStopping):
        default_tune_range = [
            IntegerSuggestion("n_components", 4, 256),
            LogUniformSuggestion("item_alpha", 1e-9, 1e-2),
            LogUniformSuggestion("user_alpha", 1e-9, 1e-2),
            CategoricalSuggestion("loss", ["bpr", "warp"]),
        ]
        recommender_class = BPRFMRecommender

        @classmethod
        def tune_range_given_memory_budget(
            cls, X: InteractionMatrix, memory_budget: int
        ) -> List[Suggestion]:
            # memory usage will be roughly 4 (float) * (n_users + n_items) * k
            n_components = _get_maximal_n_components_for_budget(X, memory_budget, 300)
            return [
                IntegerSuggestion("n_components", 4, n_components),
            ]

    _add_docstring(BPRFMOptimizer, _BaseOptimizerWithEarlyStoppingArgsString)


except:
    pass


try:
    from ..recommenders.multvae import MultVAERecommender

    class MultVAEOptimizer(BaseOptimizerWithEarlyStopping):
        default_tune_range = [
            CategoricalSuggestion("dim_z", [32, 64, 128, 256]),
            CategoricalSuggestion("enc_hidden_dims", [128, 256, 512]),
            CategoricalSuggestion("kl_anneal_goal", [0.1, 0.2, 0.4]),
        ]
        recommender_class = MultVAERecommender

        @classmethod
        def tune_range_given_memory_budget(
            cls, X: InteractionMatrix, memory_budget: int
        ) -> List[Suggestion]:
            if memory_budget * 1e6 > (X.shape[1] * 2048 * 8):
                raise LowMemoryError(
                    f"Memory budget {memory_budget} too small for MultVAE to work."
                )

            return []

    _add_docstring(MultVAEOptimizer, _BaseOptimizerWithEarlyStoppingArgsString)


except:
    warnings.warn("MultVAEOptimizer is not available.")
    pass
