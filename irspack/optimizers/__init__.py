from irspack.optimizers._optimizers import (
    AsymmetricCosineKNNOptimizer,
    AsymmetricCosineUserKNNOptimizer,
    BaseOptimizer,
    BaseOptimizerWithEarlyStopping,
    CosineKNNOptimizer,
    CosineUserKNNOptimizer,
    DenseSLIMOptimizer,
    IALSOptimizer,
    JaccardKNNOptimizer,
    NMFOptimizer,
    P3alphaOptimizer,
    RP3betaOptimizer,
    SLIMOptimizer,
    TopPopOptimizer,
    TruncatedSVDOptimizer,
    TverskyIndexKNNOptimizer,
)
from irspack.optimizers.autopilot import autopilot
from irspack.optimizers.base_optimizer import get_optimizer_class

__all__ = [
    "BaseOptimizer",
    "BaseOptimizerWithEarlyStopping",
    "TopPopOptimizer",
    "IALSOptimizer",
    "DenseSLIMOptimizer",
    "SLIMOptimizer",
    "P3alphaOptimizer",
    "RP3betaOptimizer",
    "CosineKNNOptimizer",
    "AsymmetricCosineKNNOptimizer",
    "CosineUserKNNOptimizer",
    "AsymmetricCosineUserKNNOptimizer",
    "JaccardKNNOptimizer",
    "TverskyIndexKNNOptimizer",
    "NMFOptimizer",
    "TruncatedSVDOptimizer",
    "get_optimizer_class",
    "autopilot",
]

try:
    from ._optimizers import BPRFMOptimizer

    __all__.append("BPRFMOptimizer")
except ImportError:
    pass

try:
    from ._optimizers import MultVAEOptimizer

    __all__.append("MultVAEOptimizer")
except ImportError:
    pass
