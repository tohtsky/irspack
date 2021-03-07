from typing import Optional, Tuple

import numpy as np
import pandas as pd


def split_last_n_interaction_df(
    df: pd.DataFrame,
    user_column: str,
    timestamp_column: str,
    n_heldout: Optional[int] = None,
    heldout_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe holding out last `n_heldout` or
    last `heldout_ratio` part of interactions of the users.

    Args:
        df:
            The Dataframe to be split.
        user_column :
            The column name for users.
        timestamp_column :
            The column name for "timestamp" (it doesn't have to be datetime).
        n_heldout :
            If not `None`, specifies the maximal number of last actions to be held-out. Defaults to None.
        heldout_ratio :
            Specifies how much of each user interaction will be held out.
            Ignored if ``n_heldout`` is present.

    Returns:
        First interactions and held-out interactions.
    """
    df_sorted = df.sort_values([user_column, timestamp_column])

    index_within_group = df_sorted.groupby(user_column).cumcount(ascending=False).values

    test_indicator: np.ndarray
    if n_heldout is not None:
        test_indicator = index_within_group < n_heldout
    else:
        n_user_appearnce = (
            df_sorted[user_column].value_counts().reindex(df_sorted[user_column].values)
        )
        test_indicator = index_within_group <= (n_user_appearnce * heldout_ratio).values
    return (
        df_sorted.iloc[np.where(~test_indicator)],
        df_sorted.iloc[np.where(test_indicator)],
    )
