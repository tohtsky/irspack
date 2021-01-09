from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse as sps
from sklearn.model_selection import train_test_split

from irspack.definitions import InteractionMatrix
from irspack.utils import rowwise_train_test_split


class UserTrainTestInteractionPair:
    """A class to hold users' train & test (if any) interactions and their ids.

    Args:
        user_ids:
            List of user ids. Its ``i``-th element should correspond to ``i``-th row
            of ``X_train``.
        X_train:
            The train part of interactions.
        X_test:
            The test part of interactions (if any).
            If ``None``, an empty matrix with the shape of ``X_train``
            will be created. Defaults to None.

    Raises:
        ValueError:
            when ``X_train`` and ``user_ids`` have inconsistent size.
        ValueError:
            when ``X_train`` and ``X_test`` have inconsistent size.

    """

    X_train: sps.csr_matrix
    """The train part of users' interactions."""
    X_test: sps.csr_matrix
    """The test part of users' interactions."""
    n_users: int
    """The number of users"""
    n_items: int
    """The number of items"""
    X_all: sps.csr_matrix
    """If ``X_test`` is not ``None``, equal to ``X_train + X_test``.Otherwise equals X_train."""

    def __init__(
        self,
        user_ids: List[Any],
        X_train: InteractionMatrix,
        X_test: Optional[InteractionMatrix],
    ):

        if len(user_ids) != X_train.shape[0]:
            raise ValueError("user_ids and X_train have different shapes.")

        if X_test is not None:
            if X_train.shape != X_test.shape:
                raise ValueError("X_train and X_test have different shapes.")
            X_test = sps.csr_matrix(X_test)
        else:
            X_test = sps.csr_matrix(X_train.shape, dtype=X_train.dtype)
        self.user_ids = [x for x in user_ids]
        self.X_train = sps.csr_matrix(X_train)
        self.X_test = X_test
        self.n_users = self.X_train.shape[0]
        self.n_items = self.X_train.shape[1]
        self.X_all = sps.csr_matrix(self.X_train + self.X_test)

    def concat(
        self, other: "UserTrainTestInteractionPair"
    ) -> "UserTrainTestInteractionPair":
        """Concatenate the users data.
        user_id will be ``self.user_ids + self.item_ids``.

        Returns:
            [type]: [description]

        ValueError:
            when ``self`` and ``other`` have unequal ``n_items``.
        """
        if self.n_items != other.n_items:
            raise ValueError("inconsistent n_items.")
        return UserTrainTestInteractionPair(
            self.user_ids + other.user_ids,
            sps.vstack([self.X_train, other.X_train], format="csr"),
            sps.vstack([self.X_test, other.X_test], format="csr"),
        )


def split_train_test_userwise(
    df_: pd.DataFrame,
    user_colname: str,
    item_colname: str,
    item_id_to_iid: Optional[Dict[Any, int]],
    heldout_ratio: float,
    rns: np.random.RandomState,
    rating_column: Optional[str] = None,
) -> UserTrainTestInteractionPair:
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
        item_id_to_iid = {id: i for i, id in enumerate(np.unique(df_[item_colname]))}
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
        random_seed=rns.randint(-(2 ** 31), 2 ** 31 - 1),
    )

    return UserTrainTestInteractionPair(user_ids, X_learn.tocsr(), X_predict.tocsr())


def split_dataframe_partial_user_holdout(
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
) -> Tuple[Dict[str, UserTrainTestInteractionPair], List[Any]]:
    """Splits the DataFrame and build an interaction matrix,
    holding out random interactions for a subset of randomly selected users
    (whom we call "validation users" and "test users").

    Args:
        df_all:
            The user-item interaction event log.
        user_column:
            The column name for user_id.
        item_column:
            The column name for movie_id.
        rating_column:
            The column name for ratings. If ``None``, the rating will be treated as
            ``1`` for all interactions. Defaults to None.
        n_val_user:
            The number of "validation users". Defaults to None.
        n_test_user:
            The number of "test users". Defaults to None.
        val_user_ratio:
            The percentage of "validation users" with respect to all users.
            Ignored when ``n_val_user`` is set. Defaults to 0.1.
        test_user_ratio:
            The percentage of "test users" with respect to all users.
            Ignored when ``n_text_user`` is set. Defaults to 0.1.
        heldout_ratio_val:
            The percentage of held-out interactions for "validation users". Defaults to 0.5.
        heldout_ratio_test:
            The percentage of held-out interactions for "test users". Defaults to 0.5.
        random_state:
            The random seed for this procedure. Defaults to 42.

    Raises:
        ValueError: When ``n_val_user + n_test_user`` is greater than the number of total users.

    Returns:
        A tuple consisting of:

            1. A dictionary with ``"train"``, ``"val"``, ``"test"`` as its keys and the
               coressponding dataset as its values.
            2. List of unique item ids (which corresponds to the columns of the datasets).
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

    train_user_row = df_train[user_column].map(train_uid_to_iid).values
    train_user_col = df_train.item_iid
    if rating_column is None:
        train_user_data = np.ones(df_train.shape[0])
    else:
        train_user_data = df_train[rating_column].values

    train_user_interactions = sps.csr_matrix(
        (train_user_data, (train_user_row, train_user_col)),
        shape=(len(train_uids), len(item_id_to_iid)),
    )

    valid_data: Dict[str, UserTrainTestInteractionPair] = dict(
        train=UserTrainTestInteractionPair(train_uids, train_user_interactions, None)
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
