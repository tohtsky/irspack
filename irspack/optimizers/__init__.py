from irspack.optimizers._optimizers import (
    AsymmetricCosineKNNOptimizer,
    AsymmetricCosineUserKNNOptimizer,
    BaseOptimizer,
    BaseOptimizerWithEarlyStopping,
    CosineKNNOptimizer,
    CosineUserKNNOptimizer,
    DenseSLIMOptimizer,
    EDLAEOptimizer,
    IALSOptimizer,
    JaccardKNNOptimizer,
    P3alphaOptimizer,
    RP3betaOptimizer,
    SLIMOptimizer,
    TopPopOptimizer,
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
    "EDLAEOptimizer",
    "SLIMOptimizer",
    "P3alphaOptimizer",
    "RP3betaOptimizer",
    "CosineKNNOptimizer",
    "AsymmetricCosineKNNOptimizer",
    "CosineUserKNNOptimizer",
    "AsymmetricCosineUserKNNOptimizer",
    "JaccardKNNOptimizer",
    "TverskyIndexKNNOptimizer",
    "get_optimizer_class",
    "autopilot",
]
try:
    from ._optimizers import NMFOptimizer, TruncatedSVDOptimizer

    __all__.extend(["NMFOptimizer", "TruncatedSVDOptimizer"])
except ImportError:
    pass  # pragma: no cover

try:
    from ._optimizers import BPRFMOptimizer

    __all__.append("BPRFMOptimizer")
except ImportError:
    pass  # pragma: no cover

try:
    from ._optimizers import MultVAEOptimizer

    __all__.append("MultVAEOptimizer")
except ImportError:
    pass  # pragma: no cover
