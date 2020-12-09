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

from typing import Any, List, Optional, Union

import numpy as np
from scipy import sparse as sps

InteractionMatrix = Union[sps.csr_matrix, sps.csc_matrix]
ProfileMatrix = Union[sps.csr_matrix, sps.csc_matrix, np.ndarray]

# wait until better numpy stub support
DenseScoreArray = np.ndarray
DenseMatrix = np.ndarray
UserIndexArray = np.ndarray


class UserDataSet(object):
    def __init__(
        self,
        user_ids: List[Any],
        X_learn: InteractionMatrix,
        X_predict: Optional[InteractionMatrix],
    ):
        # check shape
        if len(user_ids) != X_learn.shape[0]:
            raise ValueError("user_ids and X_learn have different shapes.")

        if X_predict is not None:
            if X_learn.shape != X_predict.shape:
                raise ValueError("X_learn and X_predict have different shapes.")
        self.user_ids = user_ids
        self.X_learn = X_learn
        self.X_predict = X_predict

    @property
    def n_users(self) -> int:
        return self.X_learn.shape[0]

    @property
    def n_items(self) -> int:
        return self.X_learn.shape[1]

    @property
    def X_all(self) -> InteractionMatrix:
        if self.X_predict is None:
            return self.X_learn
        return self.X_learn + self.X_predict
