from numpy.lib.ufunclike import _fix_and_maybe_deprecate_out_named_y
from examples.movielens.ml_1m_new import X_train_all
import logging
from logging import Logger
from typing import Type, Optional, List, Tuple, Dict, Any

import numpy as np
from ..utils import rowwise_train_test_split
from scipy import sparse as sps
from ..utils.nn import MLPOptimizer, MLP, MLPSearchConfig, MLPTrainingConfig

from ..recommenders.base import BaseRecommenderWithUserEmbedding
from ..definitions import (
    DenseMatrix,
    DenseScoreArray,
    InteractionMatrix,
    ProfileMatrix,
)
from ..parameter_tuning import Suggestion

from ..evaluator import Evaluator as Evaluator_hot
from ..optimizers.base_optimizer import BaseOptimizer
from ..optimizers import BPRFMOptimizer, IALSOptimizer, TruncatedSVDOptimizer
from ..recommenders import (
    BPRFMRecommender,
    IALSRecommender,
    TruncatedSVDRecommender,
)
from irspack.user_cold_start.recommenders.base import (
    BaseUserColdStartRecommender,
)


class CB2CFUserColdStartRecommender(BaseUserColdStartRecommender):
    def __init__(self, cf_rec: BaseRecommenderWithUserEmbedding, mlp: MLP):
        self.cf_rec = cf_rec
        self.mlp = mlp

    def _learn(self) -> None:
        pass

    def get_score(self, profile: sps.csr_matrix) -> DenseScoreArray:
        user_embedding: DenseMatrix = self.mlp.predict(
            profile.astype(np.float32).toarray()
        )
        return self.cf_rec.get_score_from_user_embedding(user_embedding).astype(
            np.float64
        )


class CB2CFUserOptimizerBase(object):
    recommender_class: Type[BaseRecommenderWithUserEmbedding]
    cf_optimizer_class: Type[BaseOptimizer]

    def __init__(
        self,
        X_all: InteractionMatrix,
        X_cf_train_all: InteractionMatrix,
        hot_evaluator: Evaluator_hot,
        X_profile: ProfileMatrix,
        nn_search_config: Optional[MLPSearchConfig] = None,
    ):
        self.X_all = X_all
        self.X_profile = X_profile
        self.X_cf_train_all = X_cf_train_all
        self.hot_evaluator = hot_evaluator
        if nn_search_config is None:
            nn_search_config = MLPSearchConfig()
        self.nn_search_config = nn_search_config

    @classmethod
    def split_and_optimize(
        cls,
        X_all: InteractionMatrix,
        X_profile: ProfileMatrix,
        evaluator_config: Dict[str, Any] = dict(cutoff=20, n_thread=4),
        target_metric: str = "ndcg",
        cf_split_config: Dict[str, Any] = dict(random_seed=42, test_ratio=0),
        nn_search_config: Optional[MLPSearchConfig] = None,
    ) -> Tuple[
        CB2CFUserColdStartRecommender, Dict[str, Any], MLPTrainingConfig
    ]:
        evaluator_config = dict(**evaluator_config, target_metric=target_metric)
        assert X_all.shape[0] == X_profile.shape[0]
        X_all = X_all
        X_profile = X_profile
        X_tr_, X_val_ = rowwise_train_test_split(X_all, **cf_split_config)
        hot_evaluator = Evaluator_hot(X_val_, 0, **evaluator_config)
        optimizer = cls(
            X_all,
            X_train_all,
            hot_evaluator,
            X_profile,
            nn_search_config=nn_search_config,
        )
        return optimizer.search_all()

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
        cf_suggest_overwrite: List[Suggestion] = [],
        cf_fixed_params: Dict[str, Any] = dict(),
    ) -> Tuple[
        CB2CFUserColdStartRecommender, Dict[str, Any], MLPTrainingConfig
    ]:
        if logger is not None:
            logger.info("Start learning the CB embedding.")
        recommender, best_config_recommender = self.search_embedding(
            n_trials,
            logger,
            timeout=timeout,
            suggest_overwrite=cf_suggest_overwrite,
            fixed_params=cf_fixed_params,
        )

        if logger is not None:
            logger.info("Start learning feature -> embedding map.")

        mlp, best_config_mlp = self.search_reconstruction(
            recommender,
            n_trials,
            logger=logger,
            config=reconstruction_search_config,
        )

        return (
            CB2CFUserColdStartRecommender(recommender, mlp),
            best_config_recommender,
            best_config_mlp,
        )

    def search_embedding(
        self,
        n_trials: int,
        logger: Optional[Logger] = None,
        timeout: Optional[int] = None,
        suggest_overwrite: List[Suggestion] = [],
        fixed_params: Dict[str, Any] = dict(),
    ) -> Tuple[BaseRecommenderWithUserEmbedding, Dict[str, Any]]:
        searcher = self.cf_optimizer_class(
            self.X_cf_train_all,
            self.hot_evaluator,
            metric="ndcg",
            logger=logger,
            suggest_overwrite=suggest_overwrite,
            fixed_params=fixed_params,
        )
        best_params, _ = searcher.optimize(n_trials=n_trials, timeout=timeout)
        rec = self.recommender_class(self.X_all, **best_params)
        rec.learn()
        return rec, best_params

    def search_reconstruction(
        self,
        rec: BaseRecommenderWithUserEmbedding,
        n_trials: int,
        logger: Optional[Logger] = None,
        config: Optional[MLPSearchConfig] = None,
    ) -> Tuple[MLP, MLPTrainingConfig]:
        embedding = rec.get_user_embedding()
        searcher = MLPOptimizer(self.X_profile, embedding, search_config=config)
        return searcher.search_param_fit_all(n_trials=n_trials, logger=logger)


class CB2BPRFMOptimizer(CB2CFUserOptimizerBase):
    recommender_class = BPRFMRecommender
    cf_optimizer_class = BPRFMOptimizer


class CB2TruncatedSVDOptimizer(CB2CFUserOptimizerBase):
    recommender_class = TruncatedSVDRecommender
    cf_optimizer_class = TruncatedSVDOptimizer


class CB2IALSOptimizer(CB2CFUserOptimizerBase):
    recommender_class = IALSRecommender
    cf_optimizer_class = IALSOptimizer
