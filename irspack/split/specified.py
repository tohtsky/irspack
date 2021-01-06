import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse as sps
from sklearn.model_selection import train_test_split

from irspack.definitions import InteractionMatrix

from .random import UserTrainTestInteractionPair


def holdout_specific_interactions(
    df: pd.DataFrame,
    user_column: str,
    item_column: str,
    interaction_indicator: np.ndarray,
    validatable_user_ratio_val: float = 0.2,
    validatable_user_ratio_test: float = 0.2,
    random_seed: Optional[int] = None,
) -> Tuple[List[Any], Dict[str, UserTrainTestInteractionPair]]:
    """Holds-out (part of) the interactions specified by the users.

    All the users will be split into two category:

        1. Those who have an interaction in the specified subset.
           We denote them as  "validatable" users.
        2. Those who don't.

    We split the users in 1. into three parts (train, validation, test)-users, and hold-out the specified interactions.
    The interactions of non-validatable users will be part of the train dataset.

    This split will be useful when want to:

        - recommend only part of the items (e.g., rather unpopular ones) to the users.
          In this case, the held-out interactions will be the ones with these specific items.
        - split the dataframe by a certain timepoint, and ensure that no information after
          that timepoint contaminates the training set.

    Args:
        df:
            The data source.
        user_column:
            The column name of the users.
        item_column:
            The column name of the items.
        interaction_indicator:
            Specifies where in ``df`` the held-out interactions are.
        validatable_user_ratio_val:
            The ratio of "validation-set users" in the "validatable users". Defaults to 0.2.
        validatable_user_ration_test:
            The ratio of "test-set users" in the "validatable users". Defaults to 0.2.
        random_seed:
            THe random seed used to split validatable users into three. Defaults to None.

    Returns:
        A tuple consiting of

            * The aligned list of all the items.
            * A dictionary with train/val/test user pairs.
    """
    v_user_ratio_train = 1 - validatable_user_ratio_val - validatable_user_ratio_test

    if v_user_ratio_train < -1e-10:
        raise ValueError(
            "validatable_use_ratio_val + validatable_user_ratio_test exceeds 1."
        )

    df = df[[user_column, item_column]].copy()

    unique_item_ids, item_index = np.unique(df[item_column], return_inverse=True)
    item_index_colname = str(uuid.uuid1())
    df[item_index_colname] = item_index

    flg_colname = str(uuid.uuid1())
    flg_column = np.zeros(df.shape[0])
    flg_column[interaction_indicator] = 1
    df[flg_colname] = flg_column

    validatable_users = np.unique(df[flg_column > 0][user_column])
    val_test_ratio = 1 - v_user_ratio_train
    if val_test_ratio >= 1.0:
        v_train_users = np.ndarray((0,), dtype=validatable_users.dtype)
        v_val_test_users = validatable_users
    else:
        v_train_users, v_val_test_users = train_test_split(
            validatable_users,
            test_size=val_test_ratio,
            random_state=random_seed,
        )
    v_val_users, v_test_users = train_test_split(
        v_val_test_users,
        test_size=(validatable_user_ratio_test) / (1 - v_user_ratio_train),
        random_state=random_seed,
    )
    df_train = pd.concat(
        [
            df[df[user_column].isin(v_train_users)],
            df[~df[user_column].isin(validatable_users)],
        ]
    )
    df_val = df[df[user_column].isin(v_val_users)]
    df_test = df[df[user_column].isin(v_test_users)]

    def df_to_dataset(
        df: pd.DataFrame, train: bool = False
    ) -> UserTrainTestInteractionPair:
        uid_unique, uindex = np.unique(df[user_column], return_inverse=True)
        iindex = df[item_index_colname].values
        if train:
            X = sps.csr_matrix(
                (np.ones(df.shape[0], dtype=np.float64), (uindex, iindex)),
                shape=(len(uid_unique), len(unique_item_ids)),
            )
            return UserTrainTestInteractionPair(uid_unique, X_train=X, X_test=None)
        (learn_index,) = np.where((df[flg_colname] == 0).values)
        (predict_index,) = np.where((df[flg_colname] > 0).values)
        X_learn = sps.csr_matrix(
            (
                np.ones(learn_index.shape[0], dtype=np.float64),
                (uindex[learn_index], iindex[learn_index]),
            ),
            shape=(len(uid_unique), len(unique_item_ids)),
        )
        X_predict = sps.csr_matrix(
            (
                np.ones(predict_index.shape[0], dtype=np.float64),
                (uindex[predict_index], iindex[predict_index]),
            ),
            shape=(len(uid_unique), len(unique_item_ids)),
        )
        return UserTrainTestInteractionPair(
            uid_unique, X_train=X_learn, X_test=X_predict
        )

    dataset = {
        "train": df_to_dataset(df_train, train=True),
        "val": df_to_dataset(df_val, train=False),
        "test": df_to_dataset(df_test, train=False),
    }
    return unique_item_ids, dataset
