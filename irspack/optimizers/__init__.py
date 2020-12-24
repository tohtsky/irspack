import warnings
from typing import List

from ..optimizers.base_optimizer import (
    BaseOptimizer,
    BaseOptimizerWithEarlyStopping,
    BaseOptimizerWithThreadingSupport,
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
    CosineKNNRecommender,
    DenseSLIMRecommender,
    IALSRecommender,
    JaccardKNNRecommender,
    NMFRecommender,
    P3alphaRecommender,
    RandomWalkWithRestartRecommender,
    RP3betaRecommender,
    SLIMRecommender,
    TopPopRecommender,
    TruncatedSVDRecommender,
    TverskyIndexKNNRecommender,
)

default_tune_range_knn = [
    IntegerSuggestion("top_k", 4, 1000),
    UniformSuggestion("shrinkage", 0, 1000),
    CategoricalSuggestion("feature_weighting", ["NONE", "TF_IDF", "BM_25"]),
]


class TopPopOptimizer(BaseOptimizer):
    default_tune_range: List[Suggestion] = []
    recommender_class = TopPopRecommender


class P3alphaOptimizer(BaseOptimizerWithThreadingSupport):
    default_tune_range = [
        LogUniformSuggestion("alpha", low=1e-10, high=2),
        IntegerSuggestion("top_k", low=10, high=1000),
        CategoricalSuggestion("normalize_weight", [True, False]),
    ]
    recommender_class = P3alphaRecommender


class IALSOptimizer(BaseOptimizerWithEarlyStopping, BaseOptimizerWithThreadingSupport):
    default_tune_range = [
        IntegerSuggestion("n_components", 4, 200),
        LogUniformSuggestion("alpha", 1, 50),
        LogUniformSuggestion("reg", 1e-10, 1e-2),
    ]
    recommender_class = IALSRecommender


class DenseSLIMOptimizer(BaseOptimizer):
    default_tune_range = [LogUniformSuggestion("reg", 1, 1e4)]
    recommender_class = DenseSLIMRecommender


class RP3betaOptimizer(BaseOptimizer):
    default_tune_range = [
        LogUniformSuggestion("alpha", 1e-5, 10),
        IntegerSuggestion("top_k", 2, 4000),
        LogUniformSuggestion("beta", 1e-5, 5e-1),
        CategoricalSuggestion("normalize_weight", [True, False]),
    ]
    recommender_class = RP3betaRecommender


class TruncatedSVDOptimizer(BaseOptimizer):
    default_tune_range = [IntegerSuggestion("n_components", 4, 512)]
    recommender_class = TruncatedSVDRecommender


class RandomWalkWithRestartOptimizer(BaseOptimizer):
    default_tune_range = [
        UniformSuggestion("decay", 1e-2, 9.9e-1),
        IntegerSuggestion("n_samples", 100, 2000, step=100),
        IntegerSuggestion("cutoff", 100, 2000, step=100),
    ]

    recommender_class = RandomWalkWithRestartRecommender


class SLIMOptimizer(BaseOptimizer):
    default_tune_range = [
        UniformSuggestion("alpha", 0, 1),
        LogUniformSuggestion("l1_ratio", 1e-6, 1),
    ]
    recommender_class = SLIMRecommender


class NMFOptimizer(BaseOptimizer):
    default_tune_range = [
        IntegerSuggestion("n_components", 4, 512),
        LogUniformSuggestion("alpha", 1e-10, 1e-1),
        UniformSuggestion("l1_ratio", 0, 1),
        CategoricalSuggestion("beta_loss", ["frobenius", "kullback-leibler"]),
    ]
    recommender_class = NMFRecommender


class CosineKNNOptimizer(BaseOptimizerWithThreadingSupport):
    default_tune_range = default_tune_range_knn.copy() + [
        CategoricalSuggestion("normalize", [False, True])
    ]

    recommender_class = CosineKNNRecommender


class JaccardKNNOptimizer(BaseOptimizerWithThreadingSupport):

    default_tune_range = default_tune_range_knn.copy()
    recommender_class = JaccardKNNRecommender


class TverskyIndexKNNOptimizer(BaseOptimizerWithThreadingSupport):
    default_tune_range = default_tune_range_knn.copy() + [
        UniformSuggestion("alpha", 0, 2),
        UniformSuggestion("beta", 0, 2),
    ]

    recommender_class = TverskyIndexKNNRecommender


class AsymmetricCosineKNNOptimizer(BaseOptimizerWithThreadingSupport):
    default_tune_range = default_tune_range_knn + [UniformSuggestion("alpha", 0, 1)]

    recommender_class = AsymmetricCosineKNNRecommender


try:
    from ..recommenders.bpr import BPRFMRecommender

    class BPRFMOptimizer(
        BaseOptimizerWithEarlyStopping, BaseOptimizerWithThreadingSupport
    ):
        default_tune_range = [
            IntegerSuggestion("n_components", 4, 256),
            LogUniformSuggestion("item_alpha", 1e-9, 1e-2),
            LogUniformSuggestion("user_alpha", 1e-9, 1e-2),
            CategoricalSuggestion("loss", ["bpr", "warp"]),
        ]
        recommender_class = BPRFMRecommender


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


except:
    warnings.warn("MultVAEOptimizer is not available.")
    pass
