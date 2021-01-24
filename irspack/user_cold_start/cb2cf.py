from logging import Logger
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
from scipy import sparse as sps

from irspack.evaluator import Evaluator as Evaluator_hot
from irspack.optimizers import IALSOptimizer, TruncatedSVDOptimizer
from irspack.optimizers.base_optimizer import BaseOptimizer as HotBaseOptimizer
from irspack.parameter_tuning import Suggestion
from irspack.recommenders import IALSRecommender, TruncatedSVDRecommender
from irspack.user_cold_start.recommenders.base import BaseUserColdStartRecommender
from irspack.utils.default_logger import get_default_logger

from ..definitions import DenseMatrix, DenseScoreArray, InteractionMatrix, ProfileMatrix
from ..recommenders.base import BaseRecommenderWithUserEmbedding
from ..utils import rowwise_train_test_split
from ..utils.nn import MLP, MLPOptimizer, MLPSearchConfig, MLPTrainingConfig


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
    cf_optimizer_class: Type[HotBaseOptimizer]

    def __init__(
        self,
        X_cf_train_all: InteractionMatrix,
        hot_evaluator: Evaluator_hot,
        X_profile: ProfileMatrix,
        nn_search_config: Optional[MLPSearchConfig] = None,
    ):
        assert X_cf_train_all.shape[0] == X_profile.shape[0]

        X_val_ground_truth = hot_evaluator.core.get_ground_truth()
        assert X_val_ground_truth.shape[1] == X_cf_train_all.shape[1]
        assert (
            hot_evaluator.offset + X_val_ground_truth.shape[0]
            <= X_cf_train_all.shape[0]
        )
        self.X_profile = X_profile
        self.X_cf_train_all = X_cf_train_all
        self.hot_evaluator = hot_evaluator

        # reconstruct all the available interaction
        X_all = X_cf_train_all + sps.vstack(
            [
                sps.csr_matrix(
                    (hot_evaluator.offset, X_cf_train_all.shape[1]),
                    dtype=np.float64,
                ),
                X_val_ground_truth,
                sps.csr_matrix(
                    (
                        X_cf_train_all.shape[0]
                        - hot_evaluator.offset
                        - X_val_ground_truth.shape[0],
                        X_cf_train_all.shape[1],
                    ),
                    dtype=np.float64,
                ),
            ],
            format="csr",
        )
        self.X_all = X_all

        if nn_search_config is None:
            nn_search_config = MLPSearchConfig()
        self.nn_search_config = nn_search_config

    @classmethod
    def split_and_optimize(
        cls,
        X_all: InteractionMatrix,
        X_profile: ProfileMatrix,
        cf_evaluator_config: Dict[str, Any] = dict(
            cutoff=20, n_threads=4, target_metric="ndcg"
        ),
        cf_split_config: Dict[str, Any] = dict(random_seed=42, test_ratio=0.2),
        nn_search_config: Optional[MLPSearchConfig] = None,
        n_trials: int = 20,
    ) -> Tuple[CB2CFUserColdStartRecommender, Dict[str, Any], MLPTrainingConfig]:
        assert X_all.shape[0] == X_profile.shape[0]
        X_all = X_all
        X_profile = X_profile
        X_tr_, X_val_ = rowwise_train_test_split(X_all, **cf_split_config)
        hot_evaluator = Evaluator_hot(X_val_, 0, **cf_evaluator_config)
        optimizer = cls(
            X_tr_,
            hot_evaluator,
            X_profile,
            nn_search_config=nn_search_config,
        )
        return optimizer.search_all(n_trials=n_trials)

    def search_all(
        self,
        n_trials: int = 40,
        logger: Optional[Logger] = None,
        timeout: Optional[int] = None,
        reconstruction_search_config: Optional[MLPSearchConfig] = None,
        cf_suggest_overwrite: List[Suggestion] = [],
        cf_fixed_params: Dict[str, Any] = dict(),
        random_seed: Optional[int] = None,
    ) -> Tuple[CB2CFUserColdStartRecommender, Dict[str, Any], MLPTrainingConfig]:
        if logger is None:
            logger = get_default_logger()
        recommender, best_config_recommender = self.search_embedding(
            n_trials,
            logger,
            timeout=timeout,
            suggest_overwrite=cf_suggest_overwrite,
            fixed_params=cf_fixed_params,
            random_seed=random_seed,
        )

        if logger is not None:
            logger.info("Start learning feature -> embedding map.")

        mlp, best_config_mlp = self.search_reconstruction(
            recommender,
            n_trials,
            logger=logger,
            config=reconstruction_search_config,
            random_seed=random_seed,
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
        random_seed: Optional[int] = None,
    ) -> Tuple[BaseRecommenderWithUserEmbedding, Dict[str, Any]]:
        searcher = self.cf_optimizer_class(
            self.X_cf_train_all,
            self.hot_evaluator,
            logger=logger,
            suggest_overwrite=suggest_overwrite,
            fixed_params=fixed_params,
        )
        best_params, _ = searcher.optimize(
            n_trials=n_trials, timeout=timeout, random_seed=random_seed
        )
        rec = self.recommender_class(self.X_all, **best_params)
        rec.learn()
        return rec, best_params

    def search_reconstruction(
        self,
        rec: BaseRecommenderWithUserEmbedding,
        n_trials: int,
        logger: Optional[Logger] = None,
        config: Optional[MLPSearchConfig] = None,
        random_seed: Optional[int] = None,
    ) -> Tuple[MLP, MLPTrainingConfig]:
        embedding = rec.get_user_embedding()
        searcher = MLPOptimizer(self.X_profile, embedding, search_config=config)
        return searcher.search_param_fit_all(
            n_trials=n_trials, logger=logger, random_seed=random_seed
        )


class CB2TruncatedSVDOptimizer(CB2CFUserOptimizerBase):
    recommender_class = TruncatedSVDRecommender
    cf_optimizer_class = TruncatedSVDOptimizer


class CB2IALSOptimizer(CB2CFUserOptimizerBase):
    recommender_class = IALSRecommender
    cf_optimizer_class = IALSOptimizer


__all__ = [
    "CB2CFUserOptimizerBase",
    "CB2IALSOptimizer",
    "CB2TruncatedSVDOptimizer",
]

try:
    from irspack.optimizers import BPRFMOptimizer
    from irspack.recommenders.bpr import BPRFMRecommender

    class CB2BPRFMOptimizer(CB2CFUserOptimizerBase):
        recommender_class = BPRFMRecommender
        cf_optimizer_class = BPRFMOptimizer

    __all__.append("CB2BPRFMOptimizer")


except:
    pass
