import warnings
from .base import (
    BaseRecommender,
    BaseSimilarityRecommender,
    BaseRecommenderWithThreadingSupport,
)
from .base_earlystop import BaseRecommenderWithEarlyStopping
from .slim import SLIMRecommender
from .mf import SVDRecommender
from .p3 import P3alphaRecommender
from .toppop import TopPopRecommender
from .rp3 import RP3betaRecommender
from .dense_slim import DenseSLIMRecommender
from .nmf import NMFRecommender
from .rwr import RandomWalkWithRestartRecommender
from .bpr import BPRFMRecommender
from .truncsvd import TruncatedSVDRecommender
from .ials import IALSRecommender

__all__ = [
    "BaseRecommender",
    "BaseSimilarityRecommender",
    "BaseRecommenderWithThreadingSupport",
    "BaseRecommenderWithEarlyStopping",
    "TopPopRecommender",
    "P3alphaRecommender",
    "RP3betaRecommender",
    "DenseSLIMRecommender",
    "NMFRecommender",
    "RandomWalkWithRestartRecommender",
    "SLIMRecommender",
    "BPRFMRecommender",
    "TruncatedSVDRecommender",
    "IALSRecommender",
    "SVDRecommender",
]

try:
    from .multvae import MultVAERecommender

    __all__.append("MultVAERecommender")
except ModuleNotFoundError:
    warnings.warn("Failed to import MultVAERecommender")
