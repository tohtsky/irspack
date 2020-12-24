from irspack.user_cold_start.optimizers import (
    BaseOptimizer as BaseUserColdStartOptimizer,
)
from irspack.user_cold_start.optimizers import (
    LinearMethodOptimizer,
    TopPopularOptimizer,
    UserCBCosineKNNOptimizer,
)
from irspack.user_cold_start.recommenders import (
    BaseUserColdStartRecommender,
    LinearMethodRecommender,
    TopPopularRecommender,
    UserCBCosineKNNRecommender,
)

from .evaluator import UserColdStartEvaluator

__all__ = [
    "BaseUserColdStartRecommender",
    "UserColdStartEvaluator",
    "LinearMethodRecommender",
    "TopPopularRecommender",
    "UserCBCosineKNNRecommender",
    "UserCBCosineKNNOptimizer",
    "TopPopularOptimizer",
    "LinearMethodOptimizer",
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
