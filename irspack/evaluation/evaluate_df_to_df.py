from typing import Any, Dict, List, Optional

import pandas as pd

from ..utils.threading import get_n_threads
from ._core import evaluate_list_vs_list


def evaluate_recommendation_df(
    df_recommendation: pd.DataFrame,
    df_ground_truth: pd.DataFrame,
    user_column: str,
    item_column: str,
    n_threads: Optional[int] = None,
) -> Dict[str, float]:
    df_recommendation = df_recommendation[
        df_recommendation[user_column].isin(df_ground_truth[user_column])
    ].copy()
    df_ground_truth = df_ground_truth[
        df_ground_truth[user_column].isin(df_recommendation[user_column])
    ].copy()
    user_ids = sorted(list(set(df_recommendation[user_column])))
    user_id_to_index: Dict[Any, int] = {uid: i for i, uid in enumerate(user_ids)}
    item_ids = sorted(
        list(set(df_recommendation[item_column]) | set(df_ground_truth[item_column]))
    )
    item_id_to_index: Dict[Any, int] = {iid: i for i, iid in enumerate(item_ids)}
    user_index_to_recommendation: Dict[int, List[int]] = {}
    user_index_to_ground_truth: Dict[int, List[int]] = {}
    for rec_row in df_recommendation.itertuples():
        uid = getattr(rec_row, user_column)
        iid = getattr(rec_row, item_column)
        u_index = user_id_to_index[uid]
        i_index = item_id_to_index[iid]
        user_index_to_recommendation.setdefault(u_index, []).append(i_index)

    for gt_row in df_recommendation.itertuples():
        uid = getattr(gt_row, user_column)
        iid = getattr(gt_row, item_column)
        u_index = user_id_to_index[uid]
        i_index = item_id_to_index[iid]
        user_index_to_ground_truth.setdefault(u_index, []).append(i_index)
    recommendation_list: List[List[int]] = []
    ground_truth_list: List[List[int]] = []
    for uindex, recommendation in user_index_to_recommendation.items():
        recommendation_list.append(recommendation)
        ground_truth_list.append(user_index_to_ground_truth.get(uindex, []))

    return evaluate_list_vs_list(
        recommendation_list,
        ground_truth_list,
        len(item_id_to_index),
        get_n_threads(n_threads),
    ).as_dict()
