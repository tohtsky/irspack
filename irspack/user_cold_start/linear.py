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
from ..definitions import InteractionMatrix
import numpy as np
from typing import Optional
from scipy import linalg
from scipy import sparse as sps

from .base import UserColdStartRecommenderBase, ProfileMatrix
from ..parameter_tuning import LogUniformSuggestion, CategoricalSuggestion


class LinearRecommender(UserColdStartRecommenderBase):
    suggest_param_range = [
        LogUniformSuggestion("lambda_", 1e-1, 1e4),
        CategoricalSuggestion("fit_intercept", [True, False]),
    ]
    W: Optional[np.ndarray]

    @classmethod
    def enlarge_profile(cls, X_profile: ProfileMatrix) -> ProfileMatrix:
        X_profile_enlarged = np.zeros(
            (X_profile.shape[0], 1 + X_profile.shape[1]), dtype=np.float32
        )
        X_profile_enlarged[:, 0] = 1
        if sps.issparse(X_profile):
            coo = X_profile.tocoo()
            X_profile_enlarged[coo.row, coo.col + 1] = coo.data
        else:
            X_profile_enlarged[:, 1:] = X_profile
        return X_profile_enlarged

    def __init__(
        self,
        X_interaction: InteractionMatrix,
        X_profile: ProfileMatrix,
        lambda_: float = 1.0,
        fit_intercept: bool = False,
    ):
        super().__init__(X_interaction, X_profile)
        self.lambda_ = lambda_
        self.fit_intercept = fit_intercept
        self.W = None

    def learn(self):
        if self.fit_intercept:
            X_profile_local = self.enlarge_profile(self.X_profile)
        else:
            X_profile_local = self.X_profile

        if sps.issparse(X_profile_local):
            X_profile_local = X_profile_local.toarray()
        X_l = X_profile_local.T.dot(X_profile_local)
        index = np.arange(X_l.shape[0])
        X_l[index, index] += self.lambda_
        inv = np.zeros_like(X_l)
        inv[index, index] = 1
        C_, lower = linalg.cho_factor(X_l, overwrite_a=True)
        inv = linalg.cho_solve((C_, lower), inv, overwrite_b=True)
        self.W = inv.dot(self.X_interaction.T.dot(X_profile_local).T)
        self.W = self.W.reshape(self.W.shape, order="F")

    def get_score(self, profile):
        if self.W is None:
            raise RuntimeError("(get_score) called before learning")
        if self.fit_intercept:
            profile = self.enlarge_profile(profile)
        if sps.issparse(profile):
            profile = profile.toarray()
        return profile.dot(self.W)
