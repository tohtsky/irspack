from irspack.user_cold_start.cb2cf import CB2BPRFMOptimizer
from .base import BaseUserColdStartRecommender
from .evaluator import UserColdStartEvaluator
from .linear import LinearRecommender
from .popular import TopPopularRecommender
from .cb_knn import UserCBKNNRecommender

__all__ = [
    "BaseUserColdStartRecommender",
    "UserColdStartEvaluator",
    "LinearRecommender",
    "TopPopularRecommender",
    "UserCBKNNRecommender",
]
try:
    from .cb2cf import CB2IALSOptimizer, CB2TruncatedSVDOptimizer

    __all__.extend(["CB2IALSOptimizer", "CB2TruncatedSVDOptimizer"])
    try:
        from .cb2cf import CB2BPRFMOptimizer

        __all__.append("CB2BPRFMOptimizer")
    except:
        pass
except:
    # must be jax-related error
    pass
