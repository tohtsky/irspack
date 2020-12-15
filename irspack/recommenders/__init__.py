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
from .truncsvd import TruncatedSVDRecommender
from .ials import IALSRecommender
from .knn import (
    CosineKNNRecommender,
    JaccardKNNRecommender,
    AsymmetricCosineKNNRecommender,
    TverskyIndexKNNRecommender,
)

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
    "TruncatedSVDRecommender",
    "IALSRecommender",
    "SVDRecommender",
    "CosineKNNRecommender",
    "JaccardKNNRecommender",
    "TverskyIndexKNNRecommender",
    "AsymmetricCosineKNNRecommender",
]

try:
    from .multvae import MultVAERecommender

    __all__.append("MultVAERecommender")
except ModuleNotFoundError:
    warnings.warn("Failed to import MultVAERecommender")

try:
    from .bpr import BPRFMRecommender

    __all__.append("BPRFMRecommender")
except:
    warnings.warn("Failed to import BPRFMRecommender")
    pass
