from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import scipy.sparse as sps

from .base import BaseEncoder
from .multi_value import ManyToManyEncoder


class DataFrameEncoder:
    def __init__(self) -> None:
        self.encoders: Dict[str, BaseEncoder] = OrderedDict()
        self.multivalue_encoders: List[Tuple[str, str, str, ManyToManyEncoder]] = []

    def add_column(self, column: str, mapper: BaseEncoder) -> "DataFrameEncoder":
        self.encoders[column] = mapper
        return self

    @property
    def encoder_shapes(self) -> List[int]:
        return [len(encoder) for encoder in self.encoders.values()] + [
            len(encoder) for _, _, _, encoder in self.multivalue_encoders
        ]

    def add_many_to_many(
        self,
        left_key: str,
        target_column: str,
        encoder: ManyToManyEncoder,
        right_key: Optional[str] = None,
    ) -> "DataFrameEncoder":
        if right_key is None:
            right_key = left_key
        self.multivalue_encoders.append((left_key, target_column, right_key, encoder))
        return self

    def transform_sparse(
        self,
        main_table: pd.DataFrame,
        many_to_many_dfs: List[pd.DataFrame] = [],
    ) -> sps.csr_matrix:

        if len(many_to_many_dfs) != len(self.multivalue_encoders):
            raise ValueError(
                f"You have to supply {len(self.multivalue_encoders)} child datafarames."
            )
        Xs: List[sps.csr_matrix] = []
        for colname, mapper in self.encoders.items():
            Xs.append(mapper.transform_sparse(main_table[colname]))

        for df_, (
            index_name_main,
            item_colname,
            index_name_child,
            encoder,
        ) in zip(many_to_many_dfs, self.multivalue_encoders):
            Xs.append(
                encoder.transform_sparse(
                    main_table,
                    df_,
                    index_name_main=index_name_main,
                    item_column_name=item_colname,
                    index_name_items=index_name_child,
                )
            )
        return sps.hstack(Xs, format="csr")
