from irspack.recommenders.base import (
    BaseRecommender,
    BaseSimilarityRecommender,
    get_recommender_class,
)
from irspack.recommenders.base_earlystop import BaseRecommenderWithEarlyStopping
from irspack.recommenders.dense_slim import DenseSLIMRecommender
from irspack.recommenders.ials import IALSRecommender
from irspack.recommenders.knn import (
    AsymmetricCosineKNNRecommender,
    CosineKNNRecommender,
    JaccardKNNRecommender,
    TverskyIndexKNNRecommender,
)
from irspack.recommenders.nmf import NMFRecommender
from irspack.recommenders.p3 import P3alphaRecommender
from irspack.recommenders.rp3 import RP3betaRecommender
from irspack.recommenders.slim import SLIMRecommender
from irspack.recommenders.toppop import TopPopRecommender
from irspack.recommenders.truncsvd import TruncatedSVDRecommender
from irspack.recommenders.user_knn import (
    AsymmetricCosineUserKNNRecommender,
    CosineUserKNNRecommender,
)

__all__ = [
    "BaseRecommender",
    "BaseSimilarityRecommender",
    "BaseRecommenderWithEarlyStopping",
    "TopPopRecommender",
    "P3alphaRecommender",
    "RP3betaRecommender",
    "DenseSLIMRecommender",
    "SLIMRecommender",
    "IALSRecommender",
    "CosineKNNRecommender",
    "JaccardKNNRecommender",
    "TverskyIndexKNNRecommender",
    "AsymmetricCosineKNNRecommender",
    "CosineUserKNNRecommender",
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
    from irspack.recommenders.multvae import MultVAERecommender

    __all__.append("MultVAERecommender")
except ImportError:  # pragma: no cover
    pass  # pragma: no cover


try:
    from irspack.recommenders.bpr import BPRFMRecommender

    __all__.append("BPRFMRecommender")
except ImportError:  # pragma: no cover
    pass  # pragma: no cover
