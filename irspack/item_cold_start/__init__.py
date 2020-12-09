from ..recommenders import TruncatedSVDRecommender, IALSRecommender, BPRFMRecommender
from ..optimizers import IALSOptimizer, TruncatedSVDOptimizer, BPRFMOptimizer
from .cb2cf import CB2CFItemOptimizerBase, CB2CFItemColdStartRecommender
from .random import RandomRecommender
from .knn import ItemCBKNNRecommender
from .evaluator import ItemColdStartEvaluator

__all__ = [
    "CB2CFTruncatedSVDOptimizer",
    "CB2CFIALSOptimizer",
    "CB2CFItemColdStartRecommender",
    "RandomRecommender",
    "ItemCBKNNRecommender",
    "ItemColdStartEvaluator",
]


class CB2CFTruncatedSVDOptimizer(CB2CFItemOptimizerBase):
    recommender_class = TruncatedSVDRecommender
    optimizer_class = TruncatedSVDOptimizer


class CB2CFIALSOptimizer(CB2CFItemOptimizerBase):
    recommender_class = IALSRecommender
    optimizer_class = IALSOptimizer


class CB2CFBPRFMOptimizer(CB2CFItemOptimizerBase):
    recommender_class = BPRFMRecommender
    optimizer_class = BPRFMOptimizer
