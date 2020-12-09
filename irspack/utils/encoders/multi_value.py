from typing import List, TypeVar, Generic, Union, Optional
import pandas as pd
import numpy as np
from scipy import sparse as sps
from collections import Counter


T = TypeVar("T")


class ManyToManyEncoder(Generic[T]):
    def __init__(self, items: List[T], normalize: bool = True, min_freq: int = 1):
        counter = Counter(items)
        self.unique_items = [item for item, cnt in counter.items() if cnt >= min_freq]
        self.names: List[Union[str, T]] = ["UNKNOWN"]
        self.names.extend(self.unique_items)
        self.normalize = normalize
        self.item_to_index = {item: i + 1 for i, item in enumerate(self.unique_items)}

    def __len__(self) -> int:
        return len(self.names)

    def transform_sparse(
        self,
        index: pd.DataFrame,
        items_df: pd.DataFrame,
        index_name_main: str,
        item_column_name: str,
        index_name_items: Optional[str] = None,
    ) -> sps.csr_matrix:
        unique_ids, inverse = np.unique(index[index_name_main], return_inverse=True)
        id_to_index = {id: i for i, id in enumerate(unique_ids)}
        relevant_items = items_df[items_df[index_name_items].isin(unique_ids)]
        row = relevant_items[index_name_items].map(id_to_index).values
        col = (
            relevant_items[item_column_name]
            .map(self.item_to_index)
            .fillna(0.0)
            .astype(np.int64)
            .values
        )
        result = sps.csr_matrix(
            (np.ones(len(row), dtype=np.float64), (row, col),),
            shape=(len(unique_ids), len(self)),
        )[inverse]
        return result
