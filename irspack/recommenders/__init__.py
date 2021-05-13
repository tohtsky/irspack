from irspack.recommenders.base import (
    BaseRecommender,
    BaseSimilarityRecommender,
    get_recommender_class,
)
from irspack.recommenders.base_earlystop import BaseRecommenderWithEarlyStopping
from irspack.recommenders.dense_slim import DenseSLIMRecommender
from irspack.recommenders.ials import IALSRecommender
from irspack.recommenders.knn import (
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
    "SLIMRecommender",
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
    "TruncatedSVDRecommender",
    "NMFRecommender",
    "get_recommender_class",
]

try:
    from irspack.recommenders.truncsvd import TruncatedSVDRecommender

    __all__.append("TruncatedSVDRecommender")
except ImportError:  # pragma: no cover
    pass  # pragma: no cover

try:
    from irspack.recommenders.nmf import NMFRecommender

    __all__.append("NMFRecommender")
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
