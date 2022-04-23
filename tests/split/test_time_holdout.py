import uuid

import numpy as np
import pandas as pd

from irspack.split import split_last_n_interaction_df

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
        ts=RNS.randint(0, 100, size=df_size).astype("datetime64[D]"),
    )
).drop_duplicates()


def test_holdout_fixed_n() -> None:
    df = df_master.copy()
    df_train, df_val = split_last_n_interaction_df(df, "user_id", "ts", 2)
    assert pd.concat([df_train, df_val]).merge(df).shape[0] == df_master.shape[0]
    uid_val_unique = np.unique(df_val.user_id)
    for uid in uid_val_unique:
        interactions_in_val = df_val[df_val.user_id == uid]
        if interactions_in_val.shape[0] == 0:
            continue
        assert interactions_in_val.shape[0] <= 2
        interactions_in_train = df_train[df_train.user_id == uid]
        if interactions_in_train.shape[0] == 0:
            continue
        assert interactions_in_train.ts.max() <= interactions_in_val.ts.min()


def test_holdout_fixed_percentage() -> None:
    df = df_master.copy()
    df_train, df_val = split_last_n_interaction_df(
        df, "user_id", "ts", heldout_ratio=0.5
    )
    assert pd.concat([df_train, df_val]).merge(df).shape[0] == df_master.shape[0]

    uid_val_unique = np.unique(df_val.user_id)
    for uid in uid_val_unique:
        interactions_in_val = df_val[df_val.user_id == uid]
        if interactions_in_val.shape[0] == 0:
            continue
        # assert interactions_in_val.shape[0] <= (0.5 * uid_cnt.loc[uid])
        interactions_in_train = df_train[df_train.user_id == uid]
        if interactions_in_train.shape[0] == 0:
            continue
        assert interactions_in_train.ts.max() <= interactions_in_val.ts.min()
