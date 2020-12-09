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
from scipy import sparse as sps


def restrict_topk_columnwise(W, top_k) -> sps.csc_matrix:
    W_temp = W.tocsc()
    data = []
    rows = []
    cols = []
    for i in range(W.shape[1]):
        data_start = W_temp.indptr[i]
        data_end = W_temp.indptr[i + 1]

        data_local = W_temp.data[data_start:data_end]
        n_restrict = min(top_k, data_end - data_start)

        local_index = data_local.argsort()[::-1][:n_restrict]

        data.append(data_local[local_index])
        rows.append(W_temp.indices[data_start:data_end][local_index])
        cols.extend([i for _ in range(n_restrict)])
    data = np.concatenate(data)
    rows = np.concatenate(rows)
    cols = np.asarray(cols)
    W = sps.csc_matrix((data, (rows, cols)), shape=W.shape)
    W.sort_indices()
    return W
