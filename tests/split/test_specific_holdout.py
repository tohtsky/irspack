import uuid

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
from numpy.core.fromnumeric import choose

from irspack.split import holdout_specific_interactions

RNS = np.random.RandomState(0)
n_users = 1000
n_items = 512
df_size = 10000

user_ids = np.asarray([str(uuid.uuid1()) for _ in range(n_users)])
item_ids = np.asarray([str(uuid.uuid1()) for _ in range(n_items)])

df_master = pd.DataFrame(
    dict(
        user_id=RNS.choice(user_ids, size=df_size, replace=True),
        item_id=RNS.choice(item_ids, size=df_size, replace=True),
    )
).drop_duplicates()

df_master["timestamp"] = RNS.randint(-1000, 1000, size=df_master.shape[0])


def test_holdout_particular_item_interaction() -> None:
    df = df_master.copy()
    validatable_item_ids = np.random.choice(
        item_ids, size=len(item_ids) // 5, replace=False
    )
    indicator = np.where(df.item_id.isin(validatable_item_ids).values)
    item_id_reprod, dataset = holdout_specific_interactions(
        df,
        "user_id",
        "item_id",
        interaction_indicator=indicator,
        validatable_user_ratio_val=0.3,
        validatable_user_ratio_test=0.3,
    )
    np.testing.assert_array_equal(item_id_reprod, np.sort(item_ids))
    item_id_to_index = {iid: i for i, iid in enumerate(item_id_reprod)}
    validatable_index = np.sort(
        np.asarray([item_id_to_index[iid] for iid in validatable_item_ids])
    )
    train = dataset["train"]
    val = dataset["val"]
    test = dataset["test"]

    assert (
        df.shape[0]
        == train.X_all.count_nonzero()
        + val.X_all.count_nonzero()
        + test.X_all.count_nonzero()
    )

    n_validatable_users_in_train = (
        (train.X_all)[:, validatable_index].sum(axis=1).A1 >= 1
    ).sum()

    total_validatable_users = n_validatable_users_in_train + val.n_users + test.n_users
    assert (
        total_validatable_users
        == np.unique(df[df.item_id.isin(validatable_item_ids)].user_id).shape[0]
    )
    assert val.n_users > 0
    assert test.n_users > 0
    assert val.X_test.sum(axis=1).A1.min() >= 1
    assert test.X_test.sum(axis=1).A1.min() >= 1

    def X_to_df(X: sps.csr_matrix, uids: np.ndarray) -> pd.DataFrame:
        row, col = X.nonzero()
        return pd.DataFrame(dict(user_id=uids[row], item_id=item_id_reprod[col]))

    assert np.all(X_to_df(val.X_test, val.user_ids).item_id.isin(validatable_item_ids))
    assert np.all(
        ~X_to_df(val.X_train, val.user_ids).item_id.isin(validatable_item_ids)
    )

    assert np.all(
        X_to_df(test.X_test, test.user_ids).item_id.isin(validatable_item_ids)
    )
    assert np.all(
        ~X_to_df(test.X_train, test.user_ids).item_id.isin(validatable_item_ids)
    )

    df_reproduced_all = pd.concat(
        [
            X_to_df(train.X_all, train.user_ids),
            X_to_df(val.X_all, val.user_ids),
            X_to_df(test.X_all, test.user_ids),
        ]
    )
    assert (
        df_reproduced_all.merge(df_master, on=["user_id", "item_id"]).shape[0]
        == df_master.shape[0]
    )

    assert 0.4 > ((n_validatable_users_in_train - 1) / total_validatable_users)
    assert 0.4 < ((n_validatable_users_in_train + 1) / total_validatable_users)
