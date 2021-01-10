import uuid
from typing import Any, List

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
    val_validatable_user_ratio = 0.3
    test_validatable_user_ratio = 0.3
    train_validatable_user_ratio = (
        1 - val_validatable_user_ratio - test_validatable_user_ratio
    )
    validatable_item_ids = np.random.choice(
        item_ids, size=len(item_ids) // 5, replace=False
    )
    indicator = np.where(df.item_id.isin(validatable_item_ids).values)
    item_id_reprod, dataset = holdout_specific_interactions(
        df,
        "user_id",
        "item_id",
        interaction_indicator=indicator,
        validatable_user_ratio_val=val_validatable_user_ratio,
        validatable_user_ratio_test=test_validatable_user_ratio,
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

    assert train_validatable_user_ratio >= (
        (n_validatable_users_in_train - 1) / total_validatable_users
    )
    assert train_validatable_user_ratio <= (
        (n_validatable_users_in_train + 1) / total_validatable_users
    )
    assert val_validatable_user_ratio <= ((val.n_users + 1) / total_validatable_users)
    assert val_validatable_user_ratio >= ((val.n_users - 1) / total_validatable_users)
    assert test_validatable_user_ratio <= ((test.n_users + 1) / total_validatable_users)
    assert test_validatable_user_ratio >= ((test.n_users - 1) / total_validatable_users)

    assert (
        total_validatable_users
        == np.unique(df[df.item_id.isin(validatable_item_ids)].user_id).shape[0]
    )
    assert val.n_users > 0
    assert test.n_users > 0
    assert val.X_test is not None
    assert val.X_test.sum(axis=1).A1.min() >= 1
    assert test.X_test is not None
    assert test.X_test.sum(axis=1).A1.min() >= 1

    def X_to_df(X: sps.csr_matrix, uids: List[Any]) -> pd.DataFrame:
        rows, cols = X.nonzero()
        return pd.DataFrame(
            dict(user_id=[uids[row] for row in rows], item_id=item_id_reprod[cols])
        )

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


def test_raise() -> None:
    df = df_master.copy()
    TS_CUTPOINT = 100
    validatable_user_ratio_val = 0.6
    validatable_user_ratio_test = 0.5
    validatable_interactions = (df.timestamp >= 0).values
    with pytest.raises(ValueError):
        unique_item_ids, dataset = holdout_specific_interactions(
            df,
            "user_id",
            "item_id",
            validatable_interactions,
            validatable_user_ratio_val=validatable_user_ratio_val,
            validatable_user_ratio_test=validatable_user_ratio_test,
            random_seed=0,
        )


def test_holdout_future() -> None:
    df = df_master.copy()
    TS_CUTPOINT = 100
    validatable_interactions = (df.timestamp >= 0).values
    validatable_user_ratio_val = 0.5
    validatable_user_ratio_test = 0.5
    unique_item_ids, dataset = holdout_specific_interactions(
        df,
        "user_id",
        "item_id",
        validatable_interactions,
        validatable_user_ratio_val=validatable_user_ratio_val,
        validatable_user_ratio_test=validatable_user_ratio_test,
        random_seed=0,
    )
    train_users = dataset["train"]
    val_users = dataset["val"]
    test_users = dataset["test"]

    df_past = df[df.timestamp < 0]
    df_future = df[df.timestamp >= 0]
    users_past_only = np.unique(
        df_past[~df_past.user_id.isin(df_future.user_id)].user_id
    )
    np.testing.assert_array_equal(train_users.user_ids, users_past_only)

    def X_to_df(X: sps.csr_matrix, uids: np.ndarray) -> pd.DataFrame:
        rows, cols = X.nonzero()
        return pd.DataFrame(
            dict(user_id=[uids[row] for row in rows], item_id=unique_item_ids[cols])
        )

    for userset in [val_users, test_users]:
        # check all of the train_interactions are in the future
        train_interactions = X_to_df(userset.X_train, userset.user_ids)
        train_interactions_with_ts = train_interactions.merge(
            df_master, on=["user_id", "item_id"], how="inner"
        )
        assert train_interactions.shape[0] == train_interactions_with_ts.shape[0]
        assert np.all(train_interactions_with_ts.timestamp.values < 0)

        # check all of the test_interactions are in the future
        test_interactions = X_to_df(userset.X_test, userset.user_ids)
        test_interactions_with_ts = test_interactions.merge(
            df_master, on=["user_id", "item_id"], how="inner"
        )

        assert test_interactions.shape[0] == test_interactions_with_ts.shape[0]
        assert np.all(test_interactions_with_ts.timestamp.values >= 0)
