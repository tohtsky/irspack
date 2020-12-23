from collections import Counter
from typing import Any, Generic, List, TypeVar, Union

import numpy as np
from scipy import sparse as sps

from .base import BaseEncoder

T = TypeVar("T")


class CategoricalValueEncoder(Generic[T], BaseEncoder):
    def __init__(self, items: List[T], min_count: int = 10):
        self.counts = Counter(items)
        relevant_items = [key for key, c in self.counts.items() if c >= min_count]
        self.item_to_index = {key: i + 1 for i, key in enumerate(relevant_items)}
        self.names: List[Union[T, str]] = []
        self.names.append("<UNK>")
        self.names.extend(relevant_items)

    def __len__(self) -> int:
        return len(self.item_to_index) + 1

    def transform_sparse(self, items: List[Any]) -> sps.csr_matrix:
        """Transform sparse"""
        cols = []
        for v in items:
            cols.append(self.item_to_index.get(v, 0))
        rows = np.arange(len(items), dtype=np.int32)
        cols = np.asarray(cols, dtype=np.int32)
        data = np.ones(rows.shape[0], dtype=np.float64)
        result = sps.csr_matrix((data, (rows, cols)), shape=(rows.shape[0], len(self)))

        return result
