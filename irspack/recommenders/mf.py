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
from .base import BaseRecommender
from ..definitions import InteractionMatrix, UserIndexArray
from scipy import sparse as sps
from sklearn.utils.extmath import randomized_svd


class SVDRecommender(BaseRecommender):
    def __init__(self, X_all: InteractionMatrix, n_components: int = 128):
        super().__init__(X_all)
        U, diag, V = randomized_svd(self.X_all, n_components=n_components)
        self.U = U
        self.V = sps.diags(diag) * V

    def get_score(self, user_indices: UserIndexArray):
        return self.U[user_indices].dot(self.V)
