from collections import Counter
from typing import List

import numpy as np
import pandas as pd
import pytest

from irspack.utils.encoders import (
    BinningEncoder,
    CategoricalValueEncoder,
    DataFrameEncoder,
    ManyToManyEncoder,
)


def test_categorical() -> None:
    CUTOFF = 2
    values = ["A", "B", "C", "A", "B"]
    cnt = Counter(values)
    valid_items = {key for key, cnt in cnt.items() if cnt >= CUTOFF}
    rare_items = {key for key, cnt in cnt.items() if cnt < CUTOFF}

    cenc = CategoricalValueEncoder[str](values, min_count=2)
    X = cenc.transform_sparse(values)
    assert X.shape[1] == len(valid_items) + 1
    assert X.sum() == len(values)
    rare_index = np.where([(x in rare_items) for x in values])
    not_rare_index = np.where([(x not in rare_items) for x in values])
    assert np.all(X[rare_index].nonzero()[1] == 0)
    assert np.all(X[not_rare_index].nonzero()[1] > 0)

    RNS = np.random.RandomState(42)
    fvalues: List[float] = RNS.randn(100)
    fenc = BinningEncoder(fvalues, n_percentiles=5)
    X = fenc.transform_sparse(np.sort(fvalues))
    assert np.all(np.diff(X.nonzero()[1]) >= 0)
    irregular_vals = np.asfarray([np.nan, -np.inf, np.inf])
    irregular_index = fenc.transform_sparse(irregular_vals).nonzero()[1]
    assert irregular_index[0] == 0  # nan's position should be 0
    assert irregular_index[1] == 1  # -inf
    assert irregular_index[2] == len(fenc) - 1

    tag_enc = ManyToManyEncoder(["tag1", "tag2", "tag3"], normalize=True)
    location_enc = ManyToManyEncoder(["loc1", "loc2"], normalize=False)
    dfenc = DataFrameEncoder().add_column("cat", cenc).add_column("price", fenc)
    dfenc.add_many_to_many(
        "item_id",
        "tag",
        tag_enc,
    )
    dfenc.add_many_to_many("item_id", "location", location_enc)

    X_all = dfenc.transform_sparse(
        pd.DataFrame(
            [
                dict(item_id=1, cat="A", price=100.0),
                dict(item_id=2, cat="B", price=0),
            ]
        ),
        [
            pd.DataFrame(
                [
                    dict(item_id=1, tag="tag1"),
                    dict(item_id=1, tag="tag2"),
                    dict(item_id=2, tag="tag3"),
                    dict(item_id=3, tag="tag3"),
                ]
            ),
            pd.DataFrame(
                [
                    dict(item_id=1, location="loc1"),
                    dict(item_id=1, location="loc1"),
                    dict(item_id=2, location="loc2"),
                ]
            ),
        ],
    )
    assert X_all.shape[1] == sum(dfenc.encoder_shapes)
    np.testing.assert_allclose(
        (
            (X_all.toarray()[:, -len(location_enc) - len(tag_enc) : -len(location_enc)])
            ** 2
        ).sum(axis=1),
        1,
    )
    np.testing.assert_allclose(
        (X_all.toarray()[:, -len(location_enc) :]).sum(axis=1),
        np.asarray([2, 1]),
    )

    X_mtom_empty = dfenc.transform_sparse(
        pd.DataFrame([dict(item_id=1, cat="A", price=100.0)]),
        [
            pd.DataFrame([dict(item_id=3, tag="")]),
            pd.DataFrame([dict(item_id=3, location="")]),
        ],
    )
    assert X_mtom_empty[:, -len(tag_enc) - len(location_enc)].count_nonzero() == 0

    with pytest.raises(ValueError):
        X_invalid_arg = dfenc.transform_sparse(
            pd.DataFrame([dict(item_id=1, cat="A", price=100.0)]),
            [
                pd.DataFrame([dict(item_id=3, tag="")]),
                # pd.DataFrame([dict(item_id=3, location="")]), # no location df
            ],
        )
