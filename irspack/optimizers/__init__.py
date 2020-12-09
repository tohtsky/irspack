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
from ..optimizers.base_optimizer import (
    BaseOptimizer,
    BaseOptimizerWithEarlyStopping,
    BaseOptimizerWithThreadingSupport,
)

from ..recommenders import (
    BPRFMRecommender,
    DenseSLIMRecommender,
    IALSRecommender,
    NMFRecommender,
    P3alphaRecommender,
    RandomWalkWithRestartRecommender,
    RP3betaRecommender,
    SLIMRecommender,
    TopPopRecommender,
    TruncatedSVDRecommender,
)


class TopPopOptimizer(BaseOptimizer):
    recommender_class = TopPopRecommender


class P3alphaOptimizer(BaseOptimizer):
    recommender_class = P3alphaRecommender


class IALSOptimizer(BaseOptimizerWithEarlyStopping, BaseOptimizerWithThreadingSupport):
    recommender_class = IALSRecommender


class BPRFMOptimizer(BaseOptimizerWithEarlyStopping, BaseOptimizerWithThreadingSupport):
    recommender_class = BPRFMRecommender


class DenseSLIMOptimizer(BaseOptimizer):
    recommender_class = DenseSLIMRecommender


class RP3betaOptimizer(BaseOptimizer):
    recommender_class = RP3betaRecommender


class TruncatedSVDOptimizer(BaseOptimizer):
    recommender_class = TruncatedSVDRecommender


class RandomWalkWithRestartOptimizer(BaseOptimizer):
    recommender_class = RandomWalkWithRestartRecommender


class SLIMOptimizer(BaseOptimizer):
    recommender_class = SLIMRecommender


class NMFOptimizer(BaseOptimizer):
    recommender_class = NMFRecommender


try:
    from ..recommenders.multvae import MultVAERecommender

    class MultVAEOptimizer(BaseOptimizerWithEarlyStopping):
        recommender_class = MultVAERecommender


except:
    warnings.warn("MultVAEOptimizer is not available.")
    pass

