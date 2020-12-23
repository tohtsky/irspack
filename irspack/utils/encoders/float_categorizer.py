from typing import List

import numpy as np
from scipy import sparse as sps

from .base import BaseEncoder


class BinningEncoder(BaseEncoder):
    def __init__(self, items: List[float], n_percentiles: int = 30):
        super().__init__()
        ps = np.linspace(0, 100, n_percentiles)
        items_farray: np.ndarray = np.asfarray(items)
        self.percentiles = np.unique(
            np.percentile(items_farray[~np.isnan(items_farray)], ps)
        )
        self.names = ["NaN"]
        self.names.append(f"(-infty, {self.percentiles[0]}]")
        for i in range(1, self.percentiles.shape[0]):
            self.names.append(f"({self.percentiles[i-1]} , {self.percentiles[i]}]")
        self.names.append(f"({self.percentiles[-1]}, infty)")

    def transform_sparse(self, items: List[str]) -> sps.csr_matrix:
        """Transform sparse"""
        items_farray: np.ndarray = np.asarray(items)
        rows = np.arange(items_farray.shape[0])
        cols = np.zeros(items_farray.shape[0], dtype=np.int32)
        valid_mask = np.where(~np.isnan(items_farray))[0]
        valid_items = items_farray[valid_mask]
        cols[valid_mask] += 1
        for v in self.percentiles:
            cols[valid_mask] += valid_items > v
        data = np.ones(rows.shape[0], dtype=np.float64)
        result = sps.csr_matrix((data, (rows, cols)), shape=(rows.shape[0], len(self)))
        assert result.shape[1] == len(self)
        return result

    def __len__(self) -> int:
        return len(self.percentiles) + 2
