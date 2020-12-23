from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse as sps
from sklearn.model_selection import train_test_split

from irspack.definitions import UserDataSet
from irspack.utils import rowwise_train_test_split


def split_train_test_userwise(
    df_: pd.DataFrame,
    user_colname: str,
    item_colname: str,
    item_id_to_iid: Optional[Dict[Any, int]],
    heldout_ratio: float,
    rns: np.random.RandomState,
    rating_column: Optional[str] = None,
) -> UserDataSet:
    """Split the user x item data frame into a pair of sparse matrix (represented as a UserDataSet).

    Parameters
    ----------
    df_ : pd.DataFrame
        user x item interaction matrix.
    user_colname : str
        The column name for the users.
    item_colname : str
        The column name for the items.
    item_id_to_iid : Optional[Dict[Any, int]]
        The mapper from item id to item index. If not supplied, create own mapping from df_.
    heldout_ratio : float
        The percentage of items (per-user) to be held out as a test(validation) ones.
    rns : np.random.RandomState
        The random state
    rating_column : Optional[str], optional
        The column for the rating values. If None, the rating values will be all equal (1), by default None

    Returns
    -------
    UserDataSet
        Resulting train-test split dataset.
    """

    if item_id_to_iid is None:
        item_id_to_iid = {
            id: i for i, id in enumerate(np.unique(df_[item_colname]))
        }
    df_ = df_[df_[item_colname].isin(item_id_to_iid.keys())]

    item_indices = df_[item_colname].map(item_id_to_iid)

    user_ids, user_indices = np.unique(df_[user_colname], return_inverse=True)
    if rating_column is not None:
        data = df_[rating_column].values
    else:
        data = np.ones(df_.shape[0], dtype=np.int32)

    X_all = sps.csr_matrix(
        (data, (user_indices, item_indices)),
        shape=(len(user_ids), len(item_id_to_iid)),
    )
    X_learn, X_predict = rowwise_train_test_split(
        X_all,
        heldout_ratio,
        random_seed=rns.randint(-(2 ** 32), 2 ** 32 - 1),
    )

    return UserDataSet(user_ids, X_learn.tocsr(), X_predict.tocsr())


def dataframe_split_user_level(
    df_all: pd.DataFrame,
    user_column: str,
    item_column: str,
    rating_column: Optional[str] = None,
    n_val_user: Optional[int] = None,
    n_test_user: Optional[int] = None,
    val_user_ratio: float = 0.1,
    test_user_ratio: float = 0.1,
    heldout_ratio_val: float = 0.5,
    heldout_ratio_test: float = 0.5,
    random_state: int = 42,
) -> Tuple[Dict[str, UserDataSet], List[Any]]:
    """
    df: contains user & item_id and rating (if any)
    user_column: column name for user_id
    item_column: column name for item_id
    rating_column: column name for rating. If this is None, all the ratings are regarded as 1.
    """
    assert (test_user_ratio <= 1) and (test_user_ratio >= 0)
    assert (val_user_ratio <= 1) and (val_user_ratio >= 0)

    uids: List[Any] = df_all[user_column].unique()
    n_users_all = len(uids)
    if n_val_user is None:
        n_val_user = int(n_users_all * val_user_ratio)
        val_user_ratio = n_val_user / n_users_all
    else:
        if n_val_user > n_users_all:
            raise ValueError("n_val_user exceeds the number of total users.")

    if n_test_user is None:
        n_test_user = int(n_users_all * test_user_ratio)
        test_user_ratio = n_test_user / n_users_all
    else:
        if (n_test_user + n_val_user) > n_users_all:
            raise ValueError(
                "n_val_user + n_test_users exceeds the number of total users."
            )

    df_all = df_all.drop_duplicates([user_column, item_column])
    rns = np.random.RandomState(random_state)
    train_uids, val_test_uids = train_test_split(
        uids, test_size=(n_val_user + n_test_user), random_state=rns
    )

    if (test_user_ratio * len(uids)) >= 1:
        val_uids, test_uids = train_test_split(
            val_test_uids,
            test_size=n_test_user,
            random_state=rns,
        )
    else:
        val_uids = val_test_uids
        test_uids = np.asarray([], dtype=val_uids.dtype)
    df_train = df_all[df_all[user_column].isin(train_uids)].copy()
    df_val = df_all[df_all[user_column].isin(val_uids)]
    df_test = df_all[df_all[user_column].isin(test_uids)]
    item_all: List[Any] = list(df_train[item_column].unique())

    def select_item(df: pd.DataFrame) -> pd.DataFrame:
        return df[df[item_column].isin(item_all)]

    df_val = select_item(df_val).copy()
    df_test = select_item(df_test).copy()
    item_id_to_iid = {id_: i for i, id_ in enumerate(item_all)}
    for df in [df_train, df_val, df_test]:
        df["item_iid"] = df[item_column].map(item_id_to_iid)

    train_uids = df_train[user_column].unique()
    train_uid_to_iid = {uid: iid for iid, uid in enumerate(train_uids)}
    X_train = sps.lil_matrix(
        (len(train_uids), len(item_id_to_iid)), dtype=(np.int32)
    )

    X_train[
        (df_train[user_column].map(train_uid_to_iid).values, df_train.item_iid)
    ] = (1 if rating_column is None else df_train[rating_column])

    valid_data: Dict[str, Any] = dict(
        train=(UserDataSet(train_uids, X_train, None))
    )

    for df_, dataset_name, heldout_ratio in [
        (df_val, "val", heldout_ratio_val),
        (df_test, "test", heldout_ratio_test),
    ]:

        valid_data[dataset_name] = split_train_test_userwise(
            df_,
            user_column,
            item_column,
            item_id_to_iid,
            heldout_ratio,
            rns,
            rating_column=rating_column,
        )
    return valid_data, item_all
