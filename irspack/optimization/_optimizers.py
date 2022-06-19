from ..parameter_tuning import (
    CategoricalSuggestion,
    IntegerSuggestion,
    UniformSuggestion,
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

#
# class TopPopOptimizer(BaseOptimizer):
#    default_tune_range: Sequence[Suggestion] = []
#    recommender_class = TopPopRecommender
#
#
# class DenseSLIMOptimizer(BaseOptimizer):
#    default_tune_range: Sequence[Suggestion] = [LogUniformSuggestion("reg", 1, 1e4)]
#    recommender_class = DenseSLIMRecommender
#
#
# class EDLAEOptimizer(BaseOptimizer):
#    default_tune_range: Sequence[Suggestion] = [
#        LogUniformSuggestion("reg", 1, 1e4),
#        UniformSuggestion("dropout_p", 0.0, 0.99),
#    ]
#    recommender_class = EDLAERecommender
#
#
# try:
#    from irspack.recommenders import NMFRecommender, TruncatedSVDRecommender
#
#    class TruncatedSVDOptimizer(BaseOptimizer):
#        default_tune_range = [IntegerSuggestion("n_components", 4, 512)]
#        recommender_class = TruncatedSVDRecommender
#
#    class NMFOptimizer(BaseOptimizer):
#        default_tune_range = [
#            IntegerSuggestion("n_components", 4, 512),
#            LogUniformSuggestion("alpha", 1e-10, 1e-1),
#            UniformSuggestion("l1_ratio", 0, 1),
#        ]
#        recommender_class = NMFRecommender
#
# except ImportError:  # pragma: no cover
#    pass  # pragma: no cover
#
#
# class SimilarityBasedOptimizerBase(BaseOptimizer):
#    pass
#
#
# class SLIMOptimizer(SimilarityBasedOptimizerBase):
#    default_tune_range = [
#        LogUniformSuggestion("alpha", 1e-5, 1),
#        UniformSuggestion("l1_ratio", 0, 1),
#    ]
#    recommender_class = SLIMRecommender
#
#
# class P3alphaOptimizer(SimilarityBasedOptimizerBase):
#    default_tune_range = [
#        IntegerSuggestion("top_k", low=10, high=1000),
#        CategoricalSuggestion("normalize_weight", [True, False]),
#    ]
#    recommender_class = P3alphaRecommender
#
#
# class RP3betaOptimizer(SimilarityBasedOptimizerBase):
#    default_tune_range = [
#        IntegerSuggestion("top_k", 2, 1000),
#        LogUniformSuggestion("beta", 1e-5, 5e-1),
#        CategoricalSuggestion("normalize_weight", [True, False]),
#    ]
#    recommender_class = RP3betaRecommender
#
#
# class CosineKNNOptimizer(SimilarityBasedOptimizerBase):
#    default_tune_range = default_tune_range_knn_with_weighting.copy() + [
#        CategoricalSuggestion("normalize", [False, True])
#    ]
#
#    recommender_class = CosineKNNRecommender
#
#
# class AsymmetricCosineKNNOptimizer(SimilarityBasedOptimizerBase):
#    default_tune_range = default_tune_range_knn_with_weighting + [
#        UniformSuggestion("alpha", 0, 1)
#    ]
#
#    recommender_class = AsymmetricCosineKNNRecommender
#
#
# class JaccardKNNOptimizer(SimilarityBasedOptimizerBase):
#
#    default_tune_range = default_tune_range_knn.copy()
#    recommender_class = JaccardKNNRecommender
#
#
# class TverskyIndexKNNOptimizer(SimilarityBasedOptimizerBase):
#    default_tune_range = default_tune_range_knn.copy() + [
#        UniformSuggestion("alpha", 0, 2),
#        UniformSuggestion("beta", 0, 2),
#    ]
#
#    recommender_class = TverskyIndexKNNRecommender
#
#
# class UserSimilarityBasedOptimizerBase(Optimizer):
#    pass
#
#
# class CosineUserKNNOptimizer(UserSimilarityBasedOptimizerBase):
#    default_tune_range = default_tune_range_knn_with_weighting.copy() + [
#        CategoricalSuggestion("normalize", [False, True])
#    ]
#
#    recommender_class = CosineUserKNNRecommender
#
#
# class AsymmetricCosineUserKNNOptimizer(UserSimilarityBasedOptimizerBase):
#    default_tune_range = default_tune_range_knn_with_weighting + [
#        UniformSuggestion("alpha", 0, 1)
#    ]
#
#    recommender_class = AsymmetricCosineUserKNNRecommender
#
#
# try:
#    from ..recommenders.bpr import BPRFMRecommender
#
#    class BPRFMOptimizer(OptimizerWithEarlyStopping):
#        default_tune_range = [
#            IntegerSuggestion("n_components", 4, 256),
#            LogUniformSuggestion("item_alpha", 1e-9, 1e-2),
#            LogUniformSuggestion("user_alpha", 1e-9, 1e-2),
#            CategoricalSuggestion("loss", ["bpr", "warp"]),
#        ]
#        recommender_class = BPRFMRecommender
#
# except:  # pragma: no cover
#    pass  # pragma: no cover
#
#
# try:
#    from ..recommenders.multvae import MultVAERecommender
#
#    class MultVAEOptimizer(OptimizerWithEarlyStopping):
#        default_tune_range = [
#            CategoricalSuggestion("dim_z", [32, 64, 128, 256]),
#            CategoricalSuggestion("enc_hidden_dims", [128, 256, 512]),
#            CategoricalSuggestion("kl_anneal_goal", [0.1, 0.2, 0.4]),
#        ]
#        recommender_class = MultVAERecommender
#
#        @classmethod
#        def tune_range_given_memory_budget(
#            cls, X: InteractionMatrix, memory_budget: int
#        ) -> Sequence[Suggestion]:
#            if memory_budget * 1e6 > (X.shape[1] * 2048 * 8):
#                raise LowMemoryError(
#                    f"Memory budget {memory_budget} too small for MultVAE to work."
#                )
#
#            return []
#
# except:  # pragma: no cover
#    pass  # pragma: no cover
#
