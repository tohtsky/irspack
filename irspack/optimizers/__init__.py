import warnings
from ..optimizers.base_optimizer import (
    BaseOptimizer,
    BaseOptimizerWithEarlyStopping,
    BaseOptimizerWithThreadingSupport,
)

from ..recommenders import (
    BPRFMRecommender,
    DenseSLIMRecommender,
    IALSRecommender,
    NMFRecommender,
    P3alphaRecommender,
    RandomWalkWithRestartRecommender,
    RP3betaRecommender,
    SLIMRecommender,
    TopPopRecommender,
    TruncatedSVDRecommender,
)


class TopPopOptimizer(BaseOptimizer):
    recommender_class = TopPopRecommender


class P3alphaOptimizer(BaseOptimizer):
    recommender_class = P3alphaRecommender


class IALSOptimizer(BaseOptimizerWithEarlyStopping, BaseOptimizerWithThreadingSupport):
    recommender_class = IALSRecommender


class BPRFMOptimizer(BaseOptimizerWithEarlyStopping, BaseOptimizerWithThreadingSupport):
    recommender_class = BPRFMRecommender


class DenseSLIMOptimizer(BaseOptimizer):
    recommender_class = DenseSLIMRecommender


class RP3betaOptimizer(BaseOptimizer):
    recommender_class = RP3betaRecommender


class TruncatedSVDOptimizer(BaseOptimizer):
    recommender_class = TruncatedSVDRecommender


class RandomWalkWithRestartOptimizer(BaseOptimizer):
    recommender_class = RandomWalkWithRestartRecommender


class SLIMOptimizer(BaseOptimizer):
    recommender_class = SLIMRecommender


class NMFOptimizer(BaseOptimizer):
    recommender_class = NMFRecommender


try:
    from ..recommenders.multvae import MultVAERecommender

    class MultVAEOptimizer(BaseOptimizerWithEarlyStopping):
        recommender_class = MultVAERecommender


except:
    warnings.warn("MultVAEOptimizer is not available.")
    pass
