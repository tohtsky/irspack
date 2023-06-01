from typing import Optional

import sklearn
from packaging import version
from sklearn.decomposition import NMF

from ..definitions import DenseScoreArray, InteractionMatrix, UserIndexArray
from ..optimization.parameter_range import (
    LogUniformFloatRange,
    UniformFloatRange,
    UniformIntegerRange,
)
from .base import BaseRecommender, RecommenderConfig


class NMFConfig(RecommenderConfig):
    n_components: int = 64
    alpha: float = 1e-2
    l1_ratio: float = 1e-2
    beta_loss: str = "frobenius"
    init: Optional[str] = None


class NMFRecommender(BaseRecommender):
    config_class = NMFConfig
    default_tune_range = [
        UniformIntegerRange("n_components", 4, 512),
        LogUniformFloatRange("alpha", 1e-10, 1e-1),
        UniformFloatRange("l1_ratio", 0, 1),
    ]

    def __init__(
        self,
        X_train_all: InteractionMatrix,
        n_components: int = 64,
        alpha: float = 1e-2,
        l1_ratio: float = 1e-2,
        beta_loss: str = "frobenius",
        init: Optional[str] = None,
    ):
        super().__init__(X_train_all)
        self.n_components = n_components
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.beta_loss = beta_loss
        self.init = init

    def _learn(self) -> None:
        # argument "alpha" in NMF was deprecated since 1.0.0
        old_version = version.parse(sklearn.__version__) < version.parse("1.0.0")
        alpha_name = "alpha" if old_version else "alpha_W"
        params = dict(
            n_components=self.n_components,
            init=self.init,
            l1_ratio=self.l1_ratio,
            beta_loss=self.beta_loss,
            random_state=42,
        )
        params[alpha_name] = self.alpha
        nmf_model = NMF(**params)
        self.nmf_model = nmf_model
        self.nmf_model.fit(self.X_train_all)
        self.W = self.nmf_model.fit_transform(self.X_train_all.tocsr())
        self.H = self.nmf_model.components_

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        res: DenseScoreArray = self.W[user_indices].dot(self.H)
        return res

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        res: DenseScoreArray = self.nmf_model.transform(X).dot(self.H)
        return res
