from .base import BaseRecommender, BaseSimilarityRecommender, get_recommender_class
from .base_earlystop import BaseRecommenderWithEarlyStopping
from .dense_slim import DenseSLIMConfig, DenseSLIMRecommender
from .edlae import EDLAEConfig, EDLAERecommender
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
from .p3 import P3alphaConfig, P3alphaRecommender
from .rp3 import RP3betaConfig, RP3betaRecommender
from .slim import SLIMConfig, SLIMRecommender
from .toppop import TopPopConfig, TopPopRecommender
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
    "EDLAERecommender",
    "EDLAEConfig",
    "SLIMConfig",
    "SLIMRecommender",
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
    from .truncsvd import TruncatedSVDConfig, TruncatedSVDRecommender

    __all__.extend(["TruncatedSVDRecommender", "TruncatedSVDConfig"])
except ImportError:  # pragma: no cover
    pass  # pragma: no cover

try:
    from .nmf import NMFConfig, NMFRecommender

    __all__.extend(["NMFRecommender", "NMFConfig"])
except ImportError:  # pragma: no cover
    pass  # pragma: no cover


try:
    from .multvae import MultVAEConfig, MultVAERecommender

    __all__.extend(["MultVAEConfig", "MultVAERecommender"])
except ImportError:  # pragma: no cover
    pass  # pragma: no cover

try:
    from .bpr import BPRFMConfig, BPRFMRecommender

    __all__.extend(["BPRFMConfig", "BPRFMRecommender"])
except ImportError:  # pragma: no cover
    pass  # pragma: no cover
