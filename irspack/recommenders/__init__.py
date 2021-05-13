import warnings

from .base import BaseRecommender, BaseSimilarityRecommender, get_recommender_class
from .base_earlystop import BaseRecommenderWithEarlyStopping
from .dense_slim import DenseSLIMConfig, DenseSLIMRecommender
from .ials import IALSConfig, IALSRecommender
from .knn import (
    AsymmetricCosineKNNConfig,
    AsymmetricCosineKNNRecommender,
    CosineKNNConfig,
    CosineKNNRecommender,
    JaccardKNNConfig,
    JaccardKNNRecommender,
    TverskyIndexKNNConfig,
    TverskyIndexKNNRecommender,
)
from .nmf import NMFConfig, NMFRecommender
from .p3 import P3alphaConfig, P3alphaRecommender
from .rp3 import RP3betaConfig, RP3betaRecommender
from .slim import SLIMConfig, SLIMRecommender
from .toppop import TopPopConfig, TopPopRecommender
from .truncsvd import TruncatedSVDConfig, TruncatedSVDRecommender
from .user_knn import (
    AsymmetricCosineUserKNNConfig,
    AsymmetricCosineUserKNNRecommender,
    CosineUserKNNConfig,
    CosineUserKNNRecommender,
)

__all__ = [
    "BaseRecommender",
    "BaseSimilarityRecommender",
    "BaseRecommenderWithEarlyStopping",
    "TopPopConfig",
    "TopPopRecommender",
    "P3alphaConfig",
    "P3alphaRecommender",
    "RP3betaConfig",
    "RP3betaRecommender",
    "DenseSLIMConfig",
    "DenseSLIMRecommender",
    "NMFConfig",
    "NMFRecommender",
    "SLIMConfig",
    "SLIMRecommender",
    "TruncatedSVDConfig",
    "TruncatedSVDRecommender",
    "IALSConfig",
    "IALSRecommender",
    "CosineKNNConfig",
    "CosineKNNRecommender",
    "JaccardKNNConfig",
    "JaccardKNNRecommender",
    "TverskyIndexKNNConfig",
    "TverskyIndexKNNRecommender",
    "AsymmetricCosineKNNConfig",
    "AsymmetricCosineKNNRecommender",
    "CosineUserKNNConfig",
    "CosineUserKNNRecommender",
    "AsymmetricCosineUserKNNConfig",
    "AsymmetricCosineUserKNNRecommender",
    "get_recommender_class",
]

try:
    from .multvae import MultVAEConfig, MultVAERecommender

    __all__.extend(["MultVAEConfig", "MultVAERecommender"])
except ModuleNotFoundError:
    warnings.warn("Failed to import MultVAERecommender")

try:
    from .bpr import BPRFMConfig, BPRFMRecommender

    __all__.extend(["BPRFMConfig", "BPRFMRecommender"])
except:
    warnings.warn(
        "Failed to import BPRFMRecommender. To enable this feature, install `lightfm`."
    )
    pass
