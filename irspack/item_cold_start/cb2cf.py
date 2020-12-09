from typing import Type, Optional, List, Dict, Any, Tuple
import logging
from logging import Logger
from .base import ItemColdStartRecommenderBase

from ..utils.nn import MLP, MLPOptimizer, MLPTrainingConfig, MLPSearchConfig
from ..optimizers import BaseOptimizer
from ..recommenders.base import BaseRecommenderWithItemEmbedding
from ..evaluator import Evaluator as Evaluator_hot
from ..definitions import (
    DenseMatrix,
    UserIndexArray,
    ProfileMatrix,
    DenseScoreArray,
    InteractionMatrix,
)
from ..utils import rowwise_train_test_split
from scipy import sparse as sps
import numpy as np
from ..parameter_tuning import Suggestion


class CB2CFItemColdStartRecommender(ItemColdStartRecommenderBase):
    def __init__(self, cf_rec: BaseRecommenderWithItemEmbedding, mlp: MLP):
        self.cf_rec = cf_rec
        self.mlp = mlp
        self.mlp.eval()

    def learn(self) -> None:
        pass

    def predict_item_embedding(self, item_profile: ProfileMatrix) -> DenseMatrix:
        return self.mlp(item_profile).detach().numpy().astype(np.float64)

    def get_score_for_user_range(
        self, user_range: UserIndexArray, item_profile: ProfileMatrix
    ) -> DenseScoreArray:
        item_embedding = self.predict_item_embedding(item_profile)
        return self.cf_rec.get_score_from_item_embedding(user_range, item_embedding)


class CB2CFItemOptimizerBase(object):
    recommender_class: Type[BaseRecommenderWithItemEmbedding]
    optimizer_class: Type[BaseOptimizer]

    def __init__(
        self,
        X_train: InteractionMatrix,
        X_profile: ProfileMatrix,
        evaluator_config=dict(cutoff=20, n_thread=4),
        target_metric: str = "ndcg",
    ):
        assert X_train.shape[1] == X_profile.shape[0]
        self.X_train = X_train
        self.X_profile = X_profile
        self.X_tr_, self.X_val_ = rowwise_train_test_split(
            X_train, random_seed=42, test_ratio=0.2
        )
        self.val_eval_local = Evaluator_hot(self.X_val_, 0, **evaluator_config)
        self.target_metric = target_metric

    def get_default_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            logger.addHandler(handler)
        return logger

    def search_all(
        self,
        n_trials: int = 40,
        logger: Optional[Logger] = None,
        timeout: Optional[int] = None,
        reconstruction_search_config: Optional[MLPSearchConfig] = None,
    ) -> Tuple[CB2CFItemColdStartRecommender, Dict[str, Any], MLPTrainingConfig]:
        if logger is not None:
            logger.info("Start learning the CB embedding.")
        recommender, best_config_recommender = self.search_embedding(
            n_trials, logger, timeout=timeout,
        )

        if logger is not None:
            logger.info("Start learning feature -> embedding map.")

        mlp, best_config_mlp = self.search_reconstruction(
            recommender, n_trials, logger=logger, config=reconstruction_search_config
        )

        return (
            CB2CFItemColdStartRecommender(recommender, mlp),
            best_config_recommender,
            best_config_mlp,
        )

    def search_embedding(
        self,
        n_trials: int,
        logger: Optional[Logger] = None,
        timeout: Optional[int] = None,
        suggest_overwrite: List[Suggestion] = [],
    ) -> Tuple[BaseRecommenderWithItemEmbedding, Dict[str, Any]]:
        searcher = self.optimizer_class(
            self.X_tr_,
            self.val_eval_local,
            metric="ndcg",
            n_trials=n_trials,
            logger=logger,
            suggest_overwrite=suggest_overwrite,
        )
        best_params, _ = searcher.do_search(self.__class__.__name__, timeout=timeout)
        rec = self.recommender_class(self.X_train, **best_params)
        rec.learn()
        return rec, best_params

    def search_reconstruction(
        self,
        rec: BaseRecommenderWithItemEmbedding,
        n_trials: int,
        logger: Optional[Logger] = None,
        config: Optional[MLPSearchConfig] = None,
    ) -> Tuple[MLP, MLPTrainingConfig]:
        embedding = rec.get_item_embedding()
        searcher = MLPOptimizer(self.X_profile, embedding, search_config=config)
        return searcher.search_param_fit_all(n_trials=n_trials, logger=logger)
