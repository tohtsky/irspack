from typing import Union
import numpy as np
from scipy import sparse as sps


def restrict_topk_columnwise(
    W: Union[sps.csr_matrix, sps.csc_matrix], top_k: int
) -> sps.csc_matrix:
    W_temp: sps.csc_matrix = W.tocsc()
    W_temp.sort_indices()
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
