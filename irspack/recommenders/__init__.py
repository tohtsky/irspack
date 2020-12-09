"""
Copyright 2020 BizReach, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
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

