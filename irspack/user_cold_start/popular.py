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
import numpy as np
from .base import UserColdStartRecommenderBase, ProfileMatrix


class TopPopularRecommender(UserColdStartRecommenderBase):
    def learn(self):
        self.popularity = self.X_interaction.sum(axis=0).astype(np.float64)

    def get_score(self, profile: ProfileMatrix):
        n_users = profile.shape[0]
        return np.repeat(self.popularity, n_users, axis=0)
