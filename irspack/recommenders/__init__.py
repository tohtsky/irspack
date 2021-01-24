import warnings

from .base import BaseRecommender, BaseSimilarityRecommender
from .base_earlystop import BaseRecommenderWithEarlyStopping
from .dense_slim import DenseSLIMRecommender
from .ials import IALSRecommender
from .knn import (
    AsymmetricCosineKNNRecommender,
    CosineKNNRecommender,
    JaccardKNNRecommender,
    TverskyIndexKNNRecommender,
)
from .nmf import NMFRecommender
from .p3 import P3alphaRecommender
from .rp3 import RP3betaRecommender
from .slim import SLIMRecommender
from .toppop import TopPopRecommender
from .truncsvd import TruncatedSVDRecommender
from .user_knn import AsymmetricCosineUserKNNRecommender, CosineUserKNNRecommender

__all__ = [
    "BaseRecommender",
    "BaseSimilarityRecommender",
    "BaseRecommenderWithEarlyStopping",
    "TopPopRecommender",
    "P3alphaRecommender",
    "RP3betaRecommender",
    "DenseSLIMRecommender",
    "NMFRecommender",
    "SLIMRecommender",
    "TruncatedSVDRecommender",
    "IALSRecommender",
    "CosineKNNRecommender",
    "JaccardKNNRecommender",
    "TverskyIndexKNNRecommender",
    "AsymmetricCosineKNNRecommender",
    "CosineUserKNNRecommender",
    "AsymmetricCosineUserKNNRecommender",
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
    warnings.warn(
        "Failed to import BPRFMRecommender. To enable this feature, install `lightfm`."
    )
    pass
