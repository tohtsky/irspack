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
    EDLAERecommender,
    IALSRecommender,
    JaccardKNNRecommender,
    P3alphaRecommender,
    RP3betaRecommender,
    SLIMRecommender,
    TopPopRecommender,
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


def _get_maximal_n_components_for_budget(
    X: InteractionMatrix, budget: int, n_component_max_default: int
) -> int:
    return min(int((budget * 1e6) / (8 * (sum(X.shape) + 1))), n_component_max_default)


class TopPopOptimizer(BaseOptimizer):
    default_tune_range: List[Suggestion] = []
    recommender_class = TopPopRecommender

    @classmethod
    def tune_range_given_memory_budget(
        cls, X: InteractionMatrix, memory_budget: int
    ) -> List[Suggestion]:
        return []


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


class DenseSLIMOptimizer(BaseOptimizer):
    default_tune_range: List[Suggestion] = [LogUniformSuggestion("reg", 1, 1e4)]
    recommender_class = DenseSLIMRecommender

    @classmethod
    def tune_range_given_memory_budget(
        cls, X: InteractionMatrix, memory_budget: int
    ) -> List[Suggestion]:
        n_items: int = X.shape[1]
        if (1e6 * memory_budget) < (4 * 2 * n_items ** 2):
            raise LowMemoryError(
                f"Memory budget {memory_budget} too small for DenseSLIM to work."
            )
        return []


class EDLAEOptimizer(BaseOptimizer):
    default_tune_range: List[Suggestion] = [
        LogUniformSuggestion("reg", 1, 1e4),
        UniformSuggestion("dropout_p", 0.0, 0.99),
    ]
    recommender_class = EDLAERecommender

    @classmethod
    def tune_range_given_memory_budget(
        cls, X: InteractionMatrix, memory_budget: int
    ) -> List[Suggestion]:
        n_items: int = X.shape[1]
        if (1e6 * memory_budget) < (4 * 2 * n_items ** 2):
            raise LowMemoryError(
                f"Memory budget {memory_budget} too small for EDLAE to work."
            )
        return []


try:
    from irspack.recommenders import NMFRecommender, TruncatedSVDRecommender

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
                IntegerSuggestion("n_components", 4, n_components),
            ]


except ImportError:  # pragma: no cover
    pass  # pragma: no cover


class SimilarityBasedOptimizerBase(BaseOptimizer):
    @classmethod
    def tune_range_given_memory_budget(
        cls, X: InteractionMatrix, memory_budget: int
    ) -> List[Suggestion]:
        top_k_max = min(int(1e6 * memory_budget / 4 // (X.shape[1] + 1)), 1024)
        if top_k_max <= 4:
            raise LowMemoryError(
                f"Memory budget {memory_budget} too small for {cls.__name__} to work."
            )

        return [
            IntegerSuggestion("top_k", 4, top_k_max),
        ]


class SLIMOptimizer(SimilarityBasedOptimizerBase):
    default_tune_range = [
        LogUniformSuggestion("alpha", 1e-5, 1),
        UniformSuggestion("l1_ratio", 0, 1),
    ]
    recommender_class = SLIMRecommender


class P3alphaOptimizer(SimilarityBasedOptimizerBase):
    default_tune_range = [
        IntegerSuggestion("top_k", low=10, high=1000),
        CategoricalSuggestion("normalize_weight", [True, False]),
    ]
    recommender_class = P3alphaRecommender


class RP3betaOptimizer(SimilarityBasedOptimizerBase):
    default_tune_range = [
        IntegerSuggestion("top_k", 2, 1000),
        LogUniformSuggestion("beta", 1e-5, 5e-1),
        CategoricalSuggestion("normalize_weight", [True, False]),
    ]
    recommender_class = RP3betaRecommender


class CosineKNNOptimizer(SimilarityBasedOptimizerBase):
    default_tune_range = default_tune_range_knn_with_weighting.copy() + [
        CategoricalSuggestion("normalize", [False, True])
    ]

    recommender_class = CosineKNNRecommender


class AsymmetricCosineKNNOptimizer(SimilarityBasedOptimizerBase):
    default_tune_range = default_tune_range_knn_with_weighting + [
        UniformSuggestion("alpha", 0, 1)
    ]

    recommender_class = AsymmetricCosineKNNRecommender


class JaccardKNNOptimizer(SimilarityBasedOptimizerBase):

    default_tune_range = default_tune_range_knn.copy()
    recommender_class = JaccardKNNRecommender


class TverskyIndexKNNOptimizer(SimilarityBasedOptimizerBase):
    default_tune_range = default_tune_range_knn.copy() + [
        UniformSuggestion("alpha", 0, 2),
        UniformSuggestion("beta", 0, 2),
    ]

    recommender_class = TverskyIndexKNNRecommender


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


class AsymmetricCosineUserKNNOptimizer(UserSimilarityBasedOptimizerBase):
    default_tune_range = default_tune_range_knn_with_weighting + [
        UniformSuggestion("alpha", 0, 1)
    ]

    recommender_class = AsymmetricCosineUserKNNRecommender


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


except:  # pragma: no cover
    pass  # pragma: no cover


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


except:  # pragma: no cover
    pass  # pragma: no cover
