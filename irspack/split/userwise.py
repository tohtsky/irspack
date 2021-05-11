from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.arraysetops import unique
from scipy import sparse as sps

from irspack.definitions import InteractionMatrix, OptionalRandomState
from irspack.split.time import split_last_n_interaction_df
from irspack.utils import df_to_sparse, rowwise_train_test_split
from irspack.utils.random import convert_randomstate


def _split_list(
    ids: List[Any], test_size: int, rns: np.random.RandomState
) -> Tuple[List[Any], List[Any]]:
    rns.shuffle(ids)
    return ids[test_size:], ids[:test_size]


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
        item_ids: Optional[List[Any]] = None,
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
        if item_ids is not None:
            if len(item_ids) != self.X_test.shape[1]:
                raise ValueError("X_train.shape[1] != len(item_ids)")
        self.item_ids = item_ids

    def _X_to_df(self, X: sps.csr_matrix, user_ids: List[Any]) -> pd.DataFrame:
        if self.item_ids is None:
            raise RuntimeError("Setting item_ids is required to use this method.")
        X.sort_indices()
        row, col = X.nonzero()
        data = X.data
        return pd.DataFrame(
            dict(
                user_id=[user_ids[r] for r in row],
                item_id=[self.item_ids[c] for c in col],
                rating=data,
            )
        )

    def df_train(self) -> pd.DataFrame:
        return self._X_to_df(self.X_train, self.user_ids)

    def df_test(self) -> pd.DataFrame:
        return self._X_to_df(self.X_test, self.user_ids)

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


def split_train_test_userwise_random(
    df_: pd.DataFrame,
    user_colname: str,
    item_colname: str,
    item_ids: List[Any],
    heldout_ratio: float,
    n_heldout: Optional[int],
    rns: np.random.RandomState,
    rating_column: Optional[str] = None,
) -> UserTrainTestInteractionPair:
    """Split the user x item data frame into a pair of sparse matrix (represented as a UserDataSet).

    Parameters
    ----------
    df_:
        user x item interaction matrix.
    user_colname:
        The column name for the users.
    item_colname:
        The column name for the items.
    item_id_to_iid:
        The mapper from item id to item index. If not supplied, create own mapping from df_.
    heldout_ratio:
        The percentage of items (per-user) to be held out as a test(validation) ones.
    n_heldout:
        The maximal number of items (per-user) to be held out as a test(validation) ones.
    rns:
        The random state
    rating_column:
        The column for the rating values. If None, the rating values will be all equal (1), by default None
    Returns
    -------
    UserDataSet
        Resulting train-test split dataset.
    """
    X_all, user_ids, _ = df_to_sparse(
        df_,
        user_colname=user_colname,
        item_colname=item_colname,
        item_ids=item_ids,
        rating_colname=rating_column,
    )

    X_learn, X_predict = rowwise_train_test_split(
        X_all,
        heldout_ratio,
        n_heldout,
        random_state=rns,
    )

    return UserTrainTestInteractionPair(
        user_ids, X_learn.tocsr(), X_predict.tocsr(), item_ids
    )


def split_train_test_userwise_time(
    df_: pd.DataFrame,
    user_colname: str,
    item_colname: str,
    time_colname: str,
    item_ids: List[Any],
    heldout_ratio: float,
    n_heldout: Optional[int],
    rating_column: Optional[str] = None,
) -> UserTrainTestInteractionPair:
    unique_user_ids = np.asarray(list(set(df_[user_colname])))
    df_train, df_test = split_last_n_interaction_df(
        df_[[user_colname, item_colname, time_colname]],
        user_colname,
        time_colname,
        n_heldout=n_heldout,
        heldout_ratio=heldout_ratio,
    )
    X_train, _, __ = df_to_sparse(
        df_train,
        user_colname,
        item_colname,
        user_ids=unique_user_ids,
        item_ids=item_ids,
        rating_colname=rating_column,
    )
    X_test, _, __ = df_to_sparse(
        df_test,
        user_colname,
        item_colname,
        user_ids=unique_user_ids,
        item_ids=item_ids,
        rating_colname=rating_column,
    )

    return UserTrainTestInteractionPair(unique_user_ids, X_train, X_test, item_ids)


def split_dataframe_partial_user_holdout(
    df_all: pd.DataFrame,
    user_column: str,
    item_column: str,
    time_column: Optional[str] = None,
    rating_column: Optional[str] = None,
    n_val_user: Optional[int] = None,
    n_test_user: Optional[int] = None,
    val_user_ratio: float = 0.1,
    test_user_ratio: float = 0.1,
    heldout_ratio_val: float = 0.5,
    n_heldout_val: Optional[int] = None,
    heldout_ratio_test: float = 0.5,
    n_heldout_test: Optional[int] = None,
    random_state: OptionalRandomState = None,
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
        time_column:
            The column name (if any) specifying the time of the interaction.
            If this is set, the split will be based on time, and some of the most recent interactions will be held out for each user.
            Defaults to None.
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
            The percentage of held-out interactions for "validation users".
            Ignored if ``n_heldout_val`` is specified. Defaults to 0.5.
        n_heldout_val:
            The maximal number of held-out interactions for "validation users".
        heldout_ratio_test:
            The percentage of held-out interactions for "test users".
            Ignored if ``n_heldout_test`` is specified. Defaults to 0.5.
        n_heldout_val:
            The maximal number of held-out interactions for "test users".
        random_state:
            The random state for this procedure. Defaults to `None`.

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
    rns = convert_randomstate(random_state)

    train_uids, val_test_uids = _split_list(uids, (n_val_user + n_test_user), rns)

    if (test_user_ratio * len(uids)) >= 1:
        val_uids, test_uids = _split_list(
            val_test_uids,
            n_test_user,
            rns,
        )
    else:
        val_uids = val_test_uids
        test_uids = np.asarray([])
    df_train = df_all[df_all[user_column].isin(train_uids)].copy()
    df_val = df_all[df_all[user_column].isin(val_uids)].copy()
    df_test = df_all[df_all[user_column].isin(test_uids)].copy()
    item_all: List[Any] = list(set(df_all[item_column]))

    train_user_interactions, _, __ = df_to_sparse(
        df_train, user_column, item_column, user_ids=train_uids, item_ids=item_all
    )

    valid_data: Dict[str, UserTrainTestInteractionPair] = dict(
        train=UserTrainTestInteractionPair(train_uids, train_user_interactions, None)
    )
    val_test_info_: List[Tuple[pd.DataFrame, str, float, Optional[int]]] = [
        (df_val, "val", heldout_ratio_val, n_heldout_val),
        (df_test, "test", heldout_ratio_test, n_heldout_test),
    ]
    for df_, dataset_name, heldout_ratio, n_heldout in val_test_info_:
        if time_column is None:
            valid_data[dataset_name] = split_train_test_userwise_random(
                df_,
                user_column,
                item_column,
                item_all,
                heldout_ratio,
                n_heldout,
                rns,
                rating_column=rating_column,
            )
        else:
            valid_data[dataset_name] = split_train_test_userwise_time(
                df_,
                user_column,
                item_column,
                time_column,
                item_all,
                heldout_ratio,
                n_heldout,
                rating_column=rating_column,
            )
    return valid_data, item_all
